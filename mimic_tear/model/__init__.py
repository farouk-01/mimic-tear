from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from data.models.tensor import TensorSchema

from .policies import Policy, PolicyInputs
from .policies.presets import PRESETS

if TYPE_CHECKING:
    from configs.models.policy import PolicyConfig

__all__ = ["Policy", "PolicyInputs", "get_policy"]


def get_policy(
    config: PolicyConfig,
    *,
    gstate_schema: TensorSchema,
    cardinalities: Mapping[str, int],
) -> Policy:
    build = PRESETS[config.preset]
    return build(config, gstate_schema=gstate_schema, cardinalities=cardinalities)
