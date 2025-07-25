import gc
import itertools
import logging
import math
import os
import signal
import typing
import warnings
from typing import Callable

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

logger = logging.getLogger(__name__)


def header(title: str | None = None, width: int = 60, fill_char: str = "-") -> str:
    if title is None:
        return fill_char * width
    title_width = len(title) + 2
    left = (width - title_width) // 2
    right = width - left - title_width
    return fill_char * left + f" {title} " + fill_char * right


def get_type_name(type_: typing.Any) -> str:
    if isinstance(type_, type):
        module = type_.__module__
        return type_.__qualname__ if module == "builtins" else f"{module}.{type_.__qualname__}"
    # Happens for aliases, None and invalid types.
    return type_


def div[T](x: T, y: T) -> T:
    """
    Ensure that numerator is divisible by the denominator and return
    the division value.
    """
    if x % y != 0:
        raise ValueError(f"{x}%{y}!=0")
    return x // y


def get_unique[T](values: typing.Iterable[T]) -> T:
    value = set(values)
    Assert.custom(lambda x: len(x) == 1, value)
    return value.pop()


def format_number(x: float | int, prec=4, exp_threshold=3) -> str:
    digits = 0 if x == 0 else math.log10(abs(x))
    if math.isfinite(digits) and -exp_threshold < math.floor(digits) < prec + exp_threshold:
        return f"{x:.{prec}f}"
    else:
        return f"{x:.{prec-1}e}"


def padded_cumsum(x: "npt.ArrayLike") -> "np.ndarray":
    import numpy as np

    y = np.hstack((0, x))
    return y.cumsum(out=y)


def clamp[T](x: T, x_min: T, x_max: T) -> T:
    return min(max(x, x_min), x_max)


def rms_diff(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    import torch

    return torch.norm(x - y, 2, dtype=torch.float32) / x.numel() ** 0.5  # noqa


class Tag:
    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return self.value

    def __deepcopy__(self, memodict: dict[str, typing.Any]) -> typing.Self:
        return self


class Assert:
    """
    A bunch of assertions that print relevant information on failure, packed into a namespace to simplify usage
    """

    @staticmethod
    def eq(x, *args, msg=None):
        for arg in args:
            assert x == arg, f"{x} != {arg} " + (f"| {msg}" if msg else "")

    @staticmethod
    def is_(x, y):
        assert x is y, f"{x} is not {y}"

    @staticmethod
    def geq(x, y):
        assert x >= y, f"{x} not >= {y}"

    @staticmethod
    def leq(x, y):
        assert x <= y, f"{x} not <= {y}"

    @staticmethod
    def gt(x, y):
        assert x > y, f"{x} not > {y}"

    @staticmethod
    def lt(x, y):
        assert x < y, f"{x} not < {y}"

    @staticmethod
    def in_range(x, low, high):
        assert low <= x < high, f"x not in range({low}, {high})"

    @staticmethod
    def in_range_incl(x, low, high):
        assert low <= x <= high, f"{x} not in (inclusive) range({low}, {high})"

    @staticmethod
    def none(x):
        assert x is None, f"Object of type {type(x)} is not None ({str(x)})"

    @staticmethod
    def empty(x):
        assert len(x) == 0, f"Not empty (len={len(x)}), {x}"

    @staticmethod
    def incl(x, y):
        assert x in y, f"{x} not in {list(y)}"

    @staticmethod
    def not_incl(x, y):
        assert x not in y, f"{x} in {y}"

    @staticmethod
    def multiple(x, y):
        assert x % y == 0, f"{x} not a multiple of {y}"

    @staticmethod
    def rms_close(x, y, threshold):
        rms = rms_diff(x, y).detach().item()
        assert rms <= threshold, f"Rms diff too big ({rms:.3e} > {threshold:.3e}) between tensors {x} and {y}"

    @staticmethod
    def rms_close_relative(x, y, threshold, min_threshold=0):
        import torch

        Assert.eq(x.shape, y.shape)
        scale = (torch.sum(x**2 + y**2) / (2 * x.numel())) ** 0.5
        threshold = max(threshold * scale, min_threshold)
        rms = rms_diff(x, y).item()
        assert rms <= threshold, f"Rms diff too big ({rms:.3e} > {threshold:.3e}) between tensors {x} and {y}"

    @staticmethod
    def all_equal(x, y):
        import torch

        # Make it work for lists and numpy arrays.
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)

        Assert.eq(x.shape, y.shape)
        neq = x != y
        if neq.any().item():  # noqa
            index = None if x.numel() == 1 else torch.where(neq)  # noqa
            raise AssertionError(
                f"Tensors have {index[0].numel()} different entries out of "
                f"{x.numel()}: {x[index]} != {y[index]} at index {torch.stack(index, -1)}"
            )

    @staticmethod
    def all_different(x, y):
        import torch

        # Make it work for numpy arrays.
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)

        eq = x == y
        if eq.any().item():  # noqa
            index = torch.where(torch.as_tensor(eq))  # noqa
            raise AssertionError(
                f"Tensors have {index[0].numel()} unexpected matching entries out of "
                f"{x.numel()}: {x[index]} != {y[index]} at index {torch.stack(index, -1)}"
            )

    @staticmethod
    def custom(fn, *args, **kwargs):
        assert fn(
            *args, **kwargs
        ), f"Assertion failed: fn({', '.join(itertools.chain((str(x) for x in args),(f'{str(k)}={str(v)}' for k,v in kwargs.items())))})"

    @staticmethod
    def not_custom(fn, *args, **kwargs):
        assert not fn(
            *args, **kwargs
        ), f"Assertion failed: not fn({', '.join(itertools.chain((str(x) for x in args),(f'{str(k)}={str(v)}' for k,v in kwargs.items())))})"


class Registry[KeyType, ValueType]:
    # TODO: Inherit from dict instead?
    def __init__(self, name: str, data: dict[KeyType, ValueType]):
        self._name = name
        self._data = data.copy()

    def __getitem__(self, key: KeyType) -> ValueType:
        if key not in self:
            raise KeyError(f"Entry {key} not found in {self._name} registry")
        return self._data[key]

    def __setitem__(self, key: KeyType, value: ValueType):
        if key in self:
            raise KeyError(f"Entry {key} already in {self._name} registry")
        self._data[key] = value

    def __delitem__(self, key: KeyType):
        if key not in self:
            raise KeyError(f"Entry {key} not found in {self._name} registry")
        del self._data[key]

    def keys(self) -> list[KeyType]:
        return list(self._data)

    def __contains__(self, key: KeyType) -> bool:
        return key in self._data

    def __iter__(self) -> typing.Iterator[KeyType]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def items(self):
        return self._data.items()

    @property
    def name(self) -> str:
        return self._name


class LazyRegistry[KeyType, ValueType](Registry[KeyType, ValueType]):
    def __getitem__(self, key: KeyType) -> ValueType:
        return super().__getitem__(key)()


def log[
    T
](*message: typing.Any, log_fn: type[BaseException] | typing.Callable[[str], T] = logger.info, join: str = ", ") -> T:
    message = join.join([str(m() if callable(m) else m) for m in message])
    logged = log_fn(message)
    if isinstance(logged, BaseException):
        raise logged
    else:
        return logged


def normalize_probabilities(p: "npt.ArrayLike", return_array: bool = False) -> "list[float] | np.ndarray":
    import numpy as np

    p = np.array(p)
    Assert.custom(lambda x: np.all(x >= 0), p)
    p_sum = p.sum()
    Assert.gt(p_sum, 0)
    out = p / p_sum
    return out if return_array else out.tolist()


class InvalidObject:
    """
    Store an error and raise it if accessed.
    Intended for missing optional imports, so that the actual import error is raised on access.
    """

    def __init__(self, error: Exception):
        self._error = error.__class__(*error.args)

    def __getattr__(self, item):
        raise self._error

    def __getitem__(self, item):

        raise self._error

    def __setitem__(self, key, value):
        raise self._error

    def __call__(self, *args, **kwargs):
        raise self._error


def try_decorate(get_decorator: Callable, _return_decorator: bool = True) -> Callable:
    """
    Try to decorate an object, but ignore the error until the object is actually used.
    The wrapped decorator should always be instantiated before calling,
    i.e.. called as `@decorator()` rather than `@decorator`.
    """

    def new_decorator(*args, **kwargs):
        try:
            out = get_decorator()(*args, **kwargs)
        except Exception as e:
            out = InvalidObject(e)
        if _return_decorator:
            return try_decorate(lambda: out, _return_decorator=False)
        return out

    return new_decorator


def compare_nested(config_a, config_b, errors: list | None = None, prefix: tuple = ()):
    if errors is None:
        errors = []
    # Check for equality of both values and types.
    if type(config_a) != type(config_b):
        errors.append(f"Type mismatch for key `{".".join(prefix)}`: {type(config_a)} != {type(config_b)}")
    if isinstance(config_a, dict):
        for key in config_a.keys() | config_b.keys():
            key_ = prefix + (key,)
            if key not in config_a:
                errors.append(f"Key `{".".join(key_)}` missing in lhs.")
            elif key not in config_b:
                errors.append(f"Key `{".".join(key_)}` missing in rhs.")
            else:
                compare_nested(config_a[key], config_b[key], errors, key_)
    elif isinstance(config_a, (list, tuple, set)):
        if len(config_a) != len(config_b):
            errors.append(f"Length mismatch for key `{".".join(prefix)}`: {len(config_a)} != {len(config_b)}.")
        else:
            for i in range(len(config_a)):
                compare_nested(config_a[i], config_b[i], errors, prefix + (str(i),))
    elif config_a != config_b and config_a is not config_b:
        # `is not` needed for special cases like `math.nan`
        errors.append(f"Different value for key `{".".join(prefix)}`: {config_a} != {config_b}.")
    return errors


def check_equal_nested(config_a, config_b):
    if errors := compare_nested(config_a, config_b):
        raise ValueError("\n".join(errors))


def get_lr_scale(
    lr_scale: float | None | tuple[float | None, ...], layer_lr_scale: float | None
) -> float | None | tuple[float | None, ...]:
    """
    Combine module and layer lr_scale.
    If one is None, return the other.
    """
    if lr_scale is None:
        return layer_lr_scale
    if layer_lr_scale is None:
        return lr_scale
    if isinstance(lr_scale, float):
        return lr_scale * layer_lr_scale
    if isinstance(lr_scale, tuple):
        return tuple(lrs * layer_lr_scale if lrs is not None else layer_lr_scale for lrs in lr_scale)
    raise ValueError(f"Invalid lr_scale: {lr_scale} (type {type(lr_scale)})")


class Interrupter:
    def __init__(self, enabled: bool = True, signals: typing.Sequence[int] = (signal.SIGINT, signal.SIGTERM)):
        self._enabled = enabled
        self._signals = signals

    def __enter__(self):
        self._interrupted = False
        self._old_signals = (
            {signum: signal.signal(signum, self._handle_signal) for signum in self._signals} if self._enabled else {}
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        for signum, handler in self._old_signals.items():
            signal.signal(signum, handler)

    def _handle_signal(self, signum, frame):
        logger.info(f"Interrupt signal {signal.Signals(signum).name} received.")
        if self._interrupted:
            # Raise for a repeated signal, ex. if a user really wants to ctrl-C.
            self._old_signals[signum](signum, frame)
        self._interrupted = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def interrupted(self):
        return self._interrupted


def set_global_variables(disable_torch_dynamo: bool = False) -> None:
    # Set global and environment variables. This needs to be called before importing any third-party package.
    # TODO: Find an alternative to get reliable tensor-parallel overlap.
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", ""):
        warnings.warn("Setting CUDA_DEVICE_MAX_CONNECTIONS breaks things.")
    # All distributed workers need the same hash seed for consistent hashing.
    if "PYTHONHASHSEED" not in os.environ:
        warnings.warn("PYTHONHASHSEED should be set and to the same value for all workers.")
    # On systems with more than 64 cores, numexpr may log an error and ignore the thread setting.
    if "NUMEXPR_MAX_THREADS" not in os.environ:
        import multiprocessing

        os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())

    if disable_torch_dynamo:
        import torch._dynamo

        torch._dynamo.config.disable = True  # noqa


_global_max_allocated = 0
_global_max_reserved = 0


def get_and_reset_memory_usage_mib(
    *,
    relative_to: dict[str, int] | None = None,
    clear_cache: bool = False,
    global_stats: bool = False,
    reset_stats: bool = True,
    reset_global_stats: bool = False,
) -> dict[str, float]:
    global _global_max_allocated, _global_max_reserved
    import torch

    if clear_cache:
        # Free memory for more accurate reporting, and to reduce OOM risk with lots of workers.
        # Cublas workspace can unnecessarily keep 100s of MBs of reserved memory.
        torch._C._cuda_clearCublasWorkspaces()
        # Lots of tensors tend to stay allocated until the next garbage collection.
        # Collect only if the remaining memory is significant enough since it's costly.
        if torch.cuda.memory_allocated() > 1e7:
            gc.collect()
        try:
            # Actually free the memory.
            torch.cuda.empty_cache()
        except RuntimeError:
            # Happens if cuda is broken.
            return {}
    report = {
        "reserved": torch.cuda.memory_reserved() / 2**20,
        "allocated": torch.cuda.memory_allocated() / 2**20,
    }
    max_allocated = torch.cuda.max_memory_allocated() / 2**20
    max_reserved = torch.cuda.max_memory_reserved() / 2**20
    if global_stats:
        report |= {
            "max_reserved": max(max_reserved, _global_max_reserved),
            "max_allocated": max(max_allocated, _global_max_allocated),
        }
    else:
        report |= {
            "max_allocated": max_allocated,
            "max_reserved": max_reserved,
            "global_max_reserved": _global_max_reserved,
        }

    if relative_to:
        report = {key: value - relative_to.get(key, 0) for key, value in report.items()}
    if reset_global_stats:
        torch.cuda.reset_peak_memory_stats()
        _global_max_reserved = 0
        _global_max_allocated = 0
    elif reset_stats:
        torch.cuda.reset_peak_memory_stats()
        _global_max_allocated = max(max_allocated, _global_max_allocated)
        _global_max_reserved = max(max_reserved, _global_max_reserved)

    return report
