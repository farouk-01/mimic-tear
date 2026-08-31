from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from graphlib import TopologicalSorter


@dataclass(frozen=True, slots=True)
class Value(ABC):
    name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Node(ABC):
    id: UUID = field(default_factory=uuid4)

    @property
    @abstractmethod
    def inputs(self) -> tuple[Value, ...]: ...

    @property
    @abstractmethod
    def outputs(self) -> tuple[Value, ...]: ...
    

@dataclass(frozen=True, slots=True)
class Plan[N: Node, V: Value]:
    inputs: tuple[V, ...]
    outputs: tuple[V, ...]
    nodes: tuple[N, ...]


class Graph:
    def __init__(self) -> None:
        self._nodes: list[Node] = []
        self._producers: dict[Value, Node] = {}
        self._consumers: dict[Value, set[Node]] = {}

    def add(self, node: Node) -> None:
        for output in node.outputs:
            if output in self._producers:
                raise ValueError(f"Value already has a producer: {output.name}")

        self._nodes.append(node)

        for output in node.outputs:
            self._producers[output] = node

        for input_ in node.inputs:
            self._consumers.setdefault(input_, set()).add(node)

    def get_producer(self, value: Value) -> Node | None:
        return self._producers.get(value)

    def get_consumers(self, value: Value) -> tuple[Node, ...]:
        return tuple(self._consumers.get(value, ()))

    def get_required_nodes(self, outputs: tuple[Value, ...]) -> set[Node]:
        required: set[Node] = set()

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

    def topological_sort(self, nodes: set[Node]) -> tuple[Node, ...]:
        dependencies: dict[Node, set[Node]] = {}

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
                producer = self.get_producer(input_)

                if producer is None and input_ not in inputs:
                    inputs.append(input_)

        return Plan(
            inputs=tuple(inputs),
            outputs=outputs,
            nodes=ordered_nodes,
        )