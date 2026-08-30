from abc import ABC, abstractmethod

from torch import Tensor

from graph.base import Node, Value, Plan


class TensorNode(Node, ABC):
    @abstractmethod
    def execute(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        ...


type TensorValues = dict[str, Tensor]


class TensorGraphExecutor:
    def execute(
        self,
        plan: Plan,
        inputs: TensorValues,
    ) -> TensorValues:
        values = inputs.copy()

        for node in plan.nodes:
            if not isinstance(node, TensorNode):
                raise TypeError(
                    f"Expected TensorNode, got {type(node).__name__}"
                )

            args = tuple(values[value.name] for value in node.inputs)
            results = node.execute(*args)

            if len(results) != len(node.outputs):
                raise ValueError(
                    "TensorNode returned a different number of values "
                    "than its declared outputs"
                )

            for output, result in zip(node.outputs, results):
                values[output.name] = result

        return {
            output.name: values[output.name]
            for output in plan.outputs
        }