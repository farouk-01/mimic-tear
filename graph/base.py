from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from graphlib import TopologicalSorter


@dataclass(frozen=True, slots=True, kw_only=True)
class Value:
    id: UUID = field(default_factory=uuid4)
    name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Node(ABC):
    id: UUID = field(default_factory=uuid4)

    @property
    @abstractmethod
    def input_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def execute(self, *inputs: object) -> tuple[object, ...]: ...


@dataclass(frozen=True, slots=True)
class BoundNode:
    node: Node
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    nodes: tuple[BoundNode, ...]


class Graph:
    def __init__(self) -> None:
        self._nodes: list[BoundNode] = []
        self._producers: dict[Value, BoundNode] = {}
        self._consumers: dict[Value, set[BoundNode]] = {}
        self._latest: dict[str, Value] = {}

    def add(self, node: Node) -> None:
        inputs = tuple(
            self._latest.setdefault(name, Value(name=name)) for name in node.input_names
        )

        outputs = tuple(Value(name=name) for name in node.output_names)

        bound = BoundNode(node=node, inputs=inputs, outputs=outputs)

        self._nodes.append(bound)

        for output in outputs:
            self._producers[output] = bound
            self._latest[output.name] = output

        for input_ in inputs:
            self._consumers.setdefault(input_, set()).add(bound)

    def value(self, name: str) -> Value:
        return self._latest.setdefault(name, Value(name=name))

    def get_producer(self, value: Value) -> BoundNode | None:
        return self._producers.get(value)

    def get_consumers(self, value: Value) -> tuple[BoundNode, ...]:
        return tuple(self._consumers.get(value, ()))

    def get_required_nodes(self, outputs: tuple[Value, ...]) -> set[BoundNode]:
        required: set[BoundNode] = set()

        def visit(value: Value) -> None:
            producer = self.get_producer(value)

            if producer is None or producer in required:
                return

            required.add(producer)

            for input_ in producer.inputs:
                visit(input_)

        for output in outputs:
            visit(output)

        return required

    def topological_sort(self, nodes: set[BoundNode]) -> tuple[BoundNode, ...]:
        dependencies: dict[BoundNode, set[BoundNode]] = {}

        for node in nodes:
            dependencies[node] = {
                producer
                for input_ in node.inputs
                if (producer := self.get_producer(input_)) is not None
                and producer in nodes
            }

        return tuple(TopologicalSorter(dependencies).static_order())

    def resolve(self, outputs: tuple[Value, ...]) -> Plan:
        required_nodes = self.get_required_nodes(outputs)
        ordered_nodes = self.topological_sort(required_nodes)

        inputs: list[Value] = []

        for output in outputs:
            if self.get_producer(output) is None and output not in inputs:
                inputs.append(output)

        for node in ordered_nodes:
            for input_ in node.inputs:
                if self.get_producer(input_) is None and input_ not in inputs:
                    inputs.append(input_)

        return Plan(
            inputs=tuple(inputs),
            outputs=outputs,
            nodes=ordered_nodes,
        )


class GraphExecutor:
    def execute(
        self,
        plan: Plan,
        inputs: dict[Value, object],
    ) -> dict[Value, object]:
        values = inputs.copy()

        for bound in plan.nodes:
            args = tuple(values[value] for value in bound.inputs)

            results = bound.node.execute(*args)

            if len(results) != len(bound.outputs):
                raise ValueError(
                    "Node returned a different number of values "
                    "than its declared outputs"
                )

            for output, result in zip(bound.outputs, results):
                values[output] = result

        return {output: values[output] for output in plan.outputs}
