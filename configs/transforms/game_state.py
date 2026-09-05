from data.process.transforms.tensor import Ratio, TensorTransform

GAME_STATE_TRANSFORMS: tuple[TensorTransform, ...] = (
    Ratio(
        output="player_hp_ratio",
        numerator="player_health",
        denominator="player_max_health",
    ),
    Ratio(
        output="player_fp_ratio",
        numerator="player_fp",
        denominator="player_max_fp",
    ),
    Ratio(
        output="player_stamina_ratio",
        numerator="player_stamina",
        denominator="player_max_stamina",
    ),
    Ratio(
        output="enemy_hp_ratio",
        numerator="enemy_health",
        denominator="enemy_max_health",
    ),
)