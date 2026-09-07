from typing import Self

from pydantic import BaseModel, ConfigDict

from data.models.tensor import TensorSchema
from data.models.gamepad import ANALOG_INPUTS, BUTTON_INPUTS
from data.process.transforms.tensor import TensorTransform
from data.process.stores.parquet import ParquetStoreConfig

from configs.transforms.controller import GAMEPAD_TRANSFORMS


class ControllerConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    tensor_schema: TensorSchema
    transforms: tuple[TensorTransform, ...] = ()
    store_cfg: ParquetStoreConfig

    @classmethod
    def load(cls, *, schema: TensorSchema) -> Self:
        transforms = GAMEPAD_TRANSFORMS
        store_cfg = ParquetStoreConfig(columns=ANALOG_INPUTS + BUTTON_INPUTS)

        return cls(
            tensor_schema=schema,
            transforms=transforms,
            store_cfg=store_cfg,
        )
