from abc import ABC, abstractmethod

from tensordict import TensorDict
from torch import Tensor

from graph.base import Node, Value, Plan

from utils import profile


class TensorNode(Node, ABC):
    @abstractmethod
    def execute(self, *inputs: Tensor) -> tuple[Tensor, ...]: ...


type TensorValues = dict[str, Tensor]


class TensorGraphExecutor:

    @profile
    def execute(
        self,
        plan: Plan,
        inputs: TensorDict,
    ) -> TensorDict:
        values: dict[Value, Tensor] = {
            value: inputs[value.name] for value in plan.inputs
        }

        for bound in plan.nodes:
            if not isinstance(bound.node, TensorNode):
                raise TypeError(f"Expected TensorNode, got {type(bound.node).__name__}")

            args = tuple(values[value] for value in bound.inputs)

            results = bound.node.execute(*args)

            if len(results) != len(bound.outputs):
                raise ValueError(
                    "TensorNode returned a different number of values "
                    "than its declared outputs"
                )

            for output, result in zip(bound.outputs, results):
                values[output] = result

        return TensorDict(
            {output.name: values[output] for output in plan.outputs},
            batch_size=inputs.batch_size,
        )
