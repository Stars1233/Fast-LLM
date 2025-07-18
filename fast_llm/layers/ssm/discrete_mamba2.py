import logging
import math

import einops
import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.tensor import ParameterMeta, init_ones_, init_uniform_, init_zeros_, kaiming_init_
from fast_llm.utils import get_lr_scale

logger = logging.getLogger(__name__)


try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _mamba_chunk_scan_combined  # noqa

    _mamba_available = True
except (ImportError, RuntimeError):
    _mamba_available = False


try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn  # noqa

    _causal_conv1d_available = True
except (ImportError, RuntimeError):
    _causal_conv1d_available = False


def bias_init_method(conv_weight):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(conv_weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    return init_uniform_(-bound, bound)


class DiscreteMamba2(torch.nn.Module):
    """DiscreteMamba2 (This code is adapted from https://github.com/cartesia-ai/edge/blob/main/cartesia-pytorch/cartesia_pytorch/Llamba/mixers/discrete_mamba2.py)."""

    def __init__(
        self,
        config: SSMConfig,
        layer_idx: int,
        tensor_space: TensorSpace,
        return_input: bool = False,
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Other options are all experimental and should not need to be configured.
        """
        # factory_kwargs = {"device": "meta"}  # , "dtype": torch.bfloat16}
        super().__init__()
        self.config: SSMConfig = config
        bias = config.add_bias_linear
        self.layer_idx = layer_idx
        self._return_input = return_input
        layer_lr_scale = config.per_layer_lr_scale[layer_idx] if config.per_layer_lr_scale else None
        mamba_layer_lr_scale = get_lr_scale(self.config.mamba_lr_scale, layer_lr_scale)
        logger.info(f"Setting lr_scale for layer {layer_idx} of type {type(self)}: {mamba_layer_lr_scale}")

        td_inner = tensor_space.get_tensor_dim(SSMDimNames.inner_dim)
        td_state = tensor_space.get_tensor_dim(SSMDimNames.state_dim)
        td_model = tensor_space.get_tensor_dim(SSMDimNames.model_dim)
        td_conv = tensor_space.get_tensor_dim(SSMDimNames.conv_dim)
        td_n_qk_heads = tensor_space.get_tensor_dim(SSMDimNames.qk_heads)
        td_n_v_heads = tensor_space.get_tensor_dim(SSMDimNames.v_heads)
        td_conv_kernel = tensor_space.get_tensor_dim(SSMDimNames.conv_kernel_size)
        td_inner_proj = tensor_space.get_tensor_dim(SSMDimNames.inner_proj_discrete_mamba2)

        self.d_model = td_model.size
        self.d_inner = td_inner.size
        self.d_state = td_state.size
        self.chunk_size = config.chunk_size
        self.n_qk_heads = td_n_qk_heads.size
        self.n_v_heads = td_n_v_heads.size
        self.conv_kernel_size = td_conv_kernel.size

        self.act = config.activation_type.activation_fn
        self.activation_name = config.activation_type.name

        # TODO: double check initializations
        # Projections
        self.in_proj = Linear(
            td_model,
            td_inner_proj,
            bias=bias,
            weight_init_method=kaiming_init_(td_model.size),
            lr_scale=mamba_layer_lr_scale,
        )
        self.z_bias = (
            ParameterMeta.from_dims(
                (td_inner,),
                weight_decay=False,
                init_method=init_zeros_,
                lr_scale=mamba_layer_lr_scale,
            )
            if not bias
            else 0.0
        )

        self.conv1d_weight = ParameterMeta.from_dims(
            (td_conv, TensorDim("1", 1), td_conv_kernel),
            init_method=init_uniform_(
                1 / math.sqrt(td_conv.size * td_conv_kernel.size), 1 / math.sqrt(td_conv.size * td_conv_kernel.size)
            ),  # see https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/conv.py#L180C53-L180C67
            lr_scale=mamba_layer_lr_scale,
        )
        self.conv1d_bias = ParameterMeta.from_dims(
            (td_conv,), init_method=bias_init_method(self.conv1d_weight), lr_scale=mamba_layer_lr_scale
        )

        # D "skip" parameter
        self.D = ParameterMeta.from_dims(
            (td_n_qk_heads,),
            weight_decay=False,
            init_method=init_ones_,
            lr_scale=mamba_layer_lr_scale,
        )

        # out_proj
        self.out_proj = Linear(
            td_inner,
            td_model,
            bias=bias,
            weight_init_method=kaiming_init_(td_inner.size),
            lr_scale=mamba_layer_lr_scale,
        )

    def forward(self, hidden_states, kwargs):
        """
        ON variable names and pep8: keeping some variable names as in the original code for clarity.

        Args:
            u: (B, L, D),

        Returns:
            outputs: dict.
            outputs["hidden_states"]: (B, L, D).
            outputs["state"]: inference cache.
        """
        if kwargs[TransformerKwargs.sequence_first]:
            raise NotImplementedError(f"Sequence-first not supported for SSMs.")

        assert _mamba_available
        input_ = hidden_states
        outputs = {}
        # assert state is None
        batch, seqlen, dim = input_.shape

        state = None

        # Hacky way to initialize state during inference
        chunk_size = self.chunk_size if state is None else seqlen

        # Pad input to nearest multiple of chunklen
        padded_len = (1 + (seqlen - 1) // chunk_size) * chunk_size
        u = torch.nn.functional.pad(input_, (0, 0, 0, padded_len - seqlen))

        # Project input
        xBCzA_log = self.in_proj(u)

        (
            xBC,
            z,
            A_log,
        ) = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        if state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead torch.nn.functional.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = einops.rearrange(xBC[:, :seqlen, :], "b l d -> b d l")
            state["conv"].copy_(
                torch.nn.functional.pad(xBC_t, (self.conv_kernel_size - xBC_t.shape[-1], 0))
            )  # Update state (B D W)

        # Convolutional layer
        xBC = self.convolutional_forward(xBC, padded_len)

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = einops.rearrange(x, "b l (h n) -> b l h n", h=self.n_v_heads)
        B = einops.rearrange(B, "b l (h n) -> b l h n", h=self.n_qk_heads)
        C = einops.rearrange(C, "b l (h n) -> b l h n", h=self.n_qk_heads)

        # SSM forward
        result = _mamba_chunk_scan_combined(
            x=x / torch.nn.functional.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=A_log,
            dt_softplus=True,
            A=-torch.ones(self.n_v_heads, device=A_log.device),
            B=B,
            C=C,
            chunk_size=chunk_size,
            # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
            return_final_states=(state is not None),
        )

        if state is not None:
            y, ssm_state = result
            state["ssm"].copy_(ssm_state)
        else:
            y = result

        Du = torch.einsum("h,blhp->blhp", self.D, x)
        y = einops.rearrange(y + Du, "b l h p -> b l (h p)")

        # Norm and gate
        out = self.out_proj(y * torch.nn.functional.silu(z + self.z_bias))
        outputs["hidden_states"] = out[:, :seqlen, :].contiguous()

        if self._return_input:
            return torch.stack([input_, outputs["hidden_states"]], dim=0)

        # TODO: since we do not support inference for now, we only return the hidden states for now.
        return outputs["hidden_states"], None

    def convolutional_forward(self, xBC, padded_len):
        """Convolutional layer forward pass for the full sequence."""
        if _causal_conv1d_available and self.activation_name in (
            "silu",
            "swish",
            "identity",
        ):
            xBC = _causal_conv1d_fn(
                xBC.transpose(1, 2),
                einops.rearrange(self.conv1d_weight, "d 1 w -> d w"),
                self.conv1d_bias,
                activation=None if self.activation_name == "identity" else self.activation_name,
            ).transpose(1, 2)
        else:
            xBC = self.act(
                torch.nn.functional.conv1d(
                    xBC.transpose(1, 2),
                    self.conv1d_weight,
                    bias=self.conv1d_bias,
                    groups=self.conv1d_weight.shape[0],
                    padding=self.conv_kernel_size - 1,
                )[..., :padded_len].transpose(1, 2)
            )
        return xBC
