import dataclasses
import functools
import itertools
import math
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, UpdateType, check_field, config_class
from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.utils import Assert, normalize_probabilities

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.indexed import ConcatenatedDataset, DatasetSlice, IndexedDataset
    from fast_llm.engine.distributed.distributed import Distributed


@config_class()
class SamplingConfig(Config):
    """
    A dataset-dependent configuration for sampling.
    """

    seed: int = Field(
        default=784569,
        desc="Seed for random sampling.",
        hint=FieldHint.feature,
    )


@dataclasses.dataclass(kw_only=True)
class SamplingParameters:
    """
    Sampling parameters set externally to the dataset and data, ex. determined by the trainer or model.
    """

    num_samples: int


@dataclasses.dataclass(kw_only=True)
class SamplingData:
    """
    Holds all the necessary information for sampling, including dataset-dependent ones (`SamplingConfig`),
    usage-dependent ones (`SamplingParameters`), and others set by the `Data`.
    """

    # TODO: Have a separate configuration (subset?) for `build`?
    config: SamplingConfig
    parameters: SamplingParameters
    cache_directory: pathlib.Path | None
    # TODO: This prevents the sampling config from being pickled in multiprocessing.
    distributed: "Distributed"
    dataset_name: str
    # Using a mutable rather than an int so it's shared with all copies made with `update`.
    _rank_counter: typing.Iterator[int] = itertools.count

    def update_config(self, update: SamplingConfig):
        return dataclasses.replace(
            self, config=self.config.from_dict(self.config, update.to_dict(), update_type=UpdateType.update)
        )

    def get_next_rank(self) -> int:
        # Counter that loops over ranks to try to distribute workloads evenly between ranks.
        return next(self._rank_counter()) % self.distributed.config.world_size


@config_class()
class DatasetConfig(Config):
    _abstract: typing.ClassVar[bool] = True


@config_class()
class SampledDatasetConfig(DatasetConfig):
    """
    A sampled dataset containing a prepared list of samples to be indexed sequentially (as-is) during training.
    """

    def build_and_sample(self, sampling: SamplingData) -> SampledDataset:
        raise NotImplementedError()


@config_class()
class SamplableDatasetConfig(SampledDatasetConfig):
    def build(self) -> SamplableDataset:
        raise NotImplementedError()

    def build_and_sample(self, sampling: SamplingData) -> SampledDataset:
        return self.build().sample(sampling)


@config_class()
class IndexedDatasetConfig(SamplableDatasetConfig):
    def _build(self) -> "IndexedDataset":
        raise NotImplementedError()


@config_class()
class ConcatenatedDatasetConfig(SamplableDatasetConfig):
    """
    Concatenate multiple indexed datasets as if they were one.
    TODO: Make a post-sampling version? (staged training)
    """

    _abstract = False
    name: str = Field(
        default="concatenated",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )
    datasets: list[IndexedDatasetConfig] = Field(
        default_factory=list,
        desc="The datasets to concatenate.",
        hint=FieldHint.core,
        valid=check_field(functools.partial(Assert.custom, lambda x: len(x) > 0)),
    )

    def build(self) -> "ConcatenatedDataset":
        from fast_llm.data.dataset.indexed import ConcatenatedDataset

        return self._build(ConcatenatedDataset)

    def _build[T: ConcatenatedDataset](self, cls: type[T]) -> T:
        return cls(self.name, [dataset.build() for dataset in self.datasets])


@config_class()
class DatasetSliceConfig(SamplableDatasetConfig):
    """
    Use a fraction of an indexed dataset, specified by the range (begin, end).
    Typically used to subsample a dataset, or to reserve part of the dataset for validation and/or testing.
    Ex. use (0.0, 0.9) for train, (0.9, 1.0) for validation for a 90%-10% split.
    TODO: This is suboptimal (duplication between train/test, unnecessary sub-datasets in the case of concatenation,
        leads to higher resource usage than necessary; more open files?)
    """

    _abstract = False
    dataset: IndexedDatasetConfig = Field(
        default=None,
        desc="The dataset to split.",
        hint=FieldHint.core,
    )
    begin: float = Field(
        default=0,
        desc="The beginning of the dataset split, as a fraction of the total samples.",
        hint=FieldHint.core,
    )
    end: float = Field(
        default=1,
        desc="The end of the dataset split, as a fraction of the total samples.",
        hint=FieldHint.core,
    )

    def build(self) -> "DatasetSlice":
        from fast_llm.data.dataset.indexed import DatasetSlice

        return self._build(DatasetSlice)

    def _build[T: DatasetSlice](self, cls: type[T]) -> T:
        dataset = self.dataset.build()
        size = len(dataset)
        return cls(
            f"{dataset.name}_{self.begin}_{self.end}",
            dataset,
            round(self.begin * size),
            round(self.end * size),
        )


@config_class()
class SampledDatasetUpdateConfig(SampledDatasetConfig):
    """
    Wrap a dataset to explicitly sample from it and optionally update its configuration parameters.
    Only explicitly set parameters (not None) will be updated, other will still be taken from `build_and_sample`'s argument.
    """

    _abstract = True
    sampling: SamplingConfig = Field(
        desc="Optional override to sampling configuration parameters.",
        hint=FieldHint.core,
    )
    dataset: SampledDatasetConfig = Field(
        desc="The dataset to sample from.",
        hint=FieldHint.core,
    )

    def build_and_sample(self, data: SamplingData) -> SampledDataset:
        return self.dataset.build_and_sample(data.update_config(self.sampling))


@config_class()
class BlendedDatasetConfig(SampledDatasetConfig):
    _abstract = False
    name: str = Field(
        default="blended",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )
    datasets: list[SampledDatasetConfig] = Field(
        default_factory=list,
        desc="The datasets to blend.",
        hint=FieldHint.core,
    )
    weights: list[float] = Field(
        default_factory=list,
        desc="The blending weight of each dataset.",
        hint=FieldHint.core,
    )
    legacy: bool = Field(
        default=False,
        desc="Use the legacy formulas for sub-dataset seeds and sample sizes.",
        hint=FieldHint.deprecated,
    )

    def _validate(self) -> None:
        self.weights = normalize_probabilities(self.weights)
        super()._validate()
        Assert.geq(len(self.datasets), 2)
        Assert.eq(len(self.datasets), len(self.weights))

    def build_and_sample(
        self,
        sampling: SamplingData,
    ) -> SampledDataset:
        from fast_llm.data.dataset.blended import BlendedDataset

        # Build and sample the datasets.

        sampled_datasets = [
            dataset.build_and_sample(
                # Blending is deterministic and the error will never be higher than 1.
                dataclasses.replace(
                    sampling,
                    parameters=dataclasses.replace(
                        sampling.parameters,
                        num_samples=(
                            math.ceil(
                                weight
                                * (
                                    sampling.parameters.num_samples
                                    + 5 * (sampling.parameters.num_samples * (1 - weight)) ** 0.5
                                )
                            )
                            if self.legacy
                            else math.ceil(weight * sampling.parameters.num_samples) + 1
                        ),
                    ),
                    # TODO: Seed may not be unique for nested blended datasets.
                    config=sampling.config.to_copy({"seed": sampling.config.seed + i * (0 if self.legacy else 697)}),
                ),
            )
            for i, (dataset, weight) in enumerate(zip(self.datasets, self.weights, strict=True))
        ]
        # Blend the datasets.
        return BlendedDataset(
            self.name,
            sampled_datasets,
            self.weights,
            sampling,
        )
