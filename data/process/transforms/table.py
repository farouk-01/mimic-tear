from typing import Any, ClassVar, cast

import polars as pl
import pyarrow as pa
from pydantic import NonNegativeInt

from data.process.transforms.base import Transform, Graph


class TableTransform(Transform[pl.Expr]):
    pass


class ButtonDilation(TableTransform):
    name: ClassVar[str] = "button_dilation"

    input: str
    before: NonNegativeInt = 0
    after: NonNegativeInt = 0
    fill_value: Any = False

    @property
    def inputs(self) -> tuple[str, ...]:
        if isinstance(self.input, str):
            return (self.input,)

        return self.input

    def __call__(self, input: pl.Expr) -> pl.Expr:
        expr = pl.max_horizontal(
            input.shift(offset, fill_value=self.fill_value)
            for offset in range(-self.before, self.after + 1)
        )

        return expr



def transform_table(table: pa.Table, transform: Graph[pl.Expr]) -> pa.Table:
    df = cast(pl.DataFrame, pl.from_arrow(table))

    cols_expr_map = {col: pl.col(col) for col in table.column_names}
    exprs = transform(cols_expr_map)
    
    return df.with_columns(**exprs).to_arrow()
