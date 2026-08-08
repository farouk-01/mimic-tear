from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnresolvedGameStateField:
    scope: str
    reason: str


# These names document the intended capture surface, but deliberately do not
# appear in GAME_STATE_VALUE_TYPES or the Parquet/model schema. Moving a field
# out of this registry requires a validated, version-resilient resolver path.
GAME_STATE_STUB_FIELDS: dict[str, UnresolvedGameStateField] = {
    "camera_yaw": UnresolvedGameStateField("dynamic", "camera chain unresolved"),
    "camera_pitch": UnresolvedGameStateField("dynamic", "camera chain unresolved"),
    "enemy_id": UnresolvedGameStateField("dynamic", "target handle chain unresolved"),
    "enemy_name": UnresolvedGameStateField("dynamic", "target identity chain unresolved"),
    "enemy_health": UnresolvedGameStateField("dynamic", "target stats chain unresolved"),
    "enemy_max_health": UnresolvedGameStateField("dynamic", "target stats chain unresolved"),
    "enemy_x": UnresolvedGameStateField("dynamic", "target transform chain unresolved"),
    "enemy_y": UnresolvedGameStateField("dynamic", "target transform chain unresolved"),
    "enemy_z": UnresolvedGameStateField("dynamic", "target transform chain unresolved"),
    "location_name": UnresolvedGameStateField("dynamic", "map-name lookup unresolved"),
    "crimson_flask_count": UnresolvedGameStateField("dynamic", "inventory chain unresolved"),
    "cerulean_flask_count": UnresolvedGameStateField("dynamic", "inventory chain unresolved"),
    "selected_quick_item_id": UnresolvedGameStateField("dynamic", "menu/equipment chain unresolved"),
    "selected_quick_item_count": UnresolvedGameStateField("dynamic", "inventory chain unresolved"),
    "quick_item_slots": UnresolvedGameStateField("dynamic", "equipment array unresolved"),
    "current_left_weapon_id": UnresolvedGameStateField("dynamic", "ChrAsm chain unresolved"),
    "current_right_weapon_id": UnresolvedGameStateField("dynamic", "ChrAsm chain unresolved"),
    "two_handing": UnresolvedGameStateField("dynamic", "weapon stance state unresolved"),
    "two_handed_weapon_id": UnresolvedGameStateField("dynamic", "weapon stance state unresolved"),
    "current_spell_id": UnresolvedGameStateField("dynamic", "spell slot chain unresolved"),
    "character_level": UnresolvedGameStateField("static", "GameDataMan chain unresolved"),
    "attribute_levels": UnresolvedGameStateField("static", "GameDataMan chain unresolved"),
    "left_weapon_slots": UnresolvedGameStateField("static", "equipment array unresolved"),
    "right_weapon_slots": UnresolvedGameStateField("static", "equipment array unresolved"),
    "weapon_upgrade_levels": UnresolvedGameStateField("static", "equipment inventory join unresolved"),
    "armor_slots": UnresolvedGameStateField("static", "equipment array unresolved"),
    "talisman_slots": UnresolvedGameStateField("static", "equipment array unresolved"),
    "spell_slots": UnresolvedGameStateField("static", "equipment array unresolved"),
    "maximum_flask_counts": UnresolvedGameStateField("static", "inventory chain unresolved"),
}
