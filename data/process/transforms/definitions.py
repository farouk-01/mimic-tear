from .generic import Ratio, Transform

GAME_STATE_TRANSFORMS: dict[str, Transform] = {
    "player_hp_ratio": Ratio("player_health", "player_max_health"),
    "player_fp_ratio": Ratio("player_fp", "player_max_fp"),
    "player_stamina_ratio": Ratio("player_stamina", "player_max_stamina"),
    "enemy_hp_ratio": Ratio("enemy_health", "enemy_max_health"),
}