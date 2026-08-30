from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
import torch

from utils import profile

type GameStateFieldKind = Literal["numeric", "categorical"]


class GameStateFieldConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    kind: GameStateFieldKind
    cardinality: int | None = None

    @model_validator(mode="before")
    def validate_cardinality(cls, values: dict) -> dict:
        kind = values.get("kind")
        cardinality = values.get("cardinality")

        if kind == "categorical" and cardinality is None:
            raise ValueError(
                f"Missing cardinality for categorical field {values.get('name')}"
            )

        if kind == "numeric" and cardinality is not None:
            raise ValueError(
                f"Unexpected cardinality for numeric field {values.get('name')}"
            )

        return values


class GameStateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fields: tuple[GameStateFieldConfig, ...] = Field(
        default_factory=tuple,
        min_length=1,
    )
    d_model: int = Field(gt=0)


class GameState(nn.Module):
    def __init__(
        self,
        *,
        fields: tuple[GameStateFieldConfig, ...],
        d_model: int,
    ) -> None:
        super().__init__()
        self.fields = fields
        self.d_model = d_model

        self.embeddings = nn.ModuleDict(
            {
                field.name: nn.Embedding(
                    num_embeddings=field.cardinality,
                    embedding_dim=d_model,
                )
                for field in fields
                if field.kind == "categorical" and field.cardinality is not None
            }
        )

        self.projections = nn.ModuleDict(
            {
                field.name: nn.Linear(1, d_model)
                for field in fields
                if field.kind == "numeric"
            }
        )

    @profile
    def forward(self, state: dict[str, Tensor]) -> Tensor:
        tokens: list[Tensor] = []

        for field in self.fields:
            tensor = state[field.name]

            if field.kind == "categorical":
                token = self.embeddings[field.name](tensor.long())

            else:
                token = self.projections[field.name](tensor.unsqueeze(-1).float())

            tokens.append(token)

        return torch.stack(tokens, dim=-2)
