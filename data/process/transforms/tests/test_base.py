from typing import ClassVar

from data.process.transforms import Graph, Transform


class AddOne(Transform[int]):
    name: ClassVar[str] = "add_one"

    input: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    def __call__(self, input: int) -> int:
        return input + 1


class Add(Transform[int]):
    name: ClassVar[str] = "add"

    lhs: str
    rhs: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.lhs, self.rhs

    def __call__(self, lhs: int, rhs: int) -> int:
        return lhs + rhs


class TestGraph:
    def test_internal_transform_dependencies_are_in_graph(self):
        transforms = (
            Add(
                output="enemy_hp_ratio",
                lhs="enemy_health",
                rhs="enemy_max_health",
            ),
            AddOne(
                output="enemy_hp_ratio_previous",
                input="enemy_hp_ratio",
            ),
            Add(
                output="enemy_damage_dealt",
                lhs="enemy_hp_ratio_previous",
                rhs="enemy_hp_ratio",
            ),
        )

        graph = Graph[int](transforms)

        assert set(graph.outputs) == {
            "enemy_hp_ratio",
            "enemy_hp_ratio_previous",
            "enemy_damage_dealt",
        }

        assert set(graph.inputs) == {
            "enemy_health",
            "enemy_max_health",
        }
