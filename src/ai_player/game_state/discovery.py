from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from ai_player.platform.windows.process_memory import (
    MemoryReadError,
    PESection,
    ProcessMemory,
)


class AddressDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BytePattern:
    values: bytes
    masks: bytes

    @classmethod
    def parse(cls, text: str) -> "BytePattern":
        if not isinstance(text, str) or not text.strip():
            raise ValueError("AOB pattern must be a non-empty string")
        values = bytearray()
        masks = bytearray()
        for token in text.split():
            if token in ("?", "??"):
                values.append(0)
                masks.append(0)
                continue
            if len(token) != 2:
                raise ValueError(f"Invalid AOB token: {token!r}")
            value = 0
            mask = 0
            for index, character in enumerate(token):
                shift = 4 if index == 0 else 0
                if character == "?":
                    continue
                try:
                    nibble = int(character, 16)
                except ValueError as error:
                    raise ValueError(f"Invalid AOB token: {token!r}") from error
                value |= nibble << shift
                mask |= 0xF << shift
            values.append(value)
            masks.append(mask)
        return cls(bytes(values), bytes(masks))

    def __len__(self) -> int:
        return len(self.values)

    def matches(self, data: bytes, offset: int) -> bool:
        if offset < 0 or offset + len(self) > len(data):
            return False
        return all(
            ((data[offset + index] ^ value) & mask) == 0
            for index, (value, mask) in enumerate(zip(self.values, self.masks))
        )

    def find_all(self, data: bytes, *, limit: int | None = None) -> list[int]:
        if limit is not None and limit <= 0:
            return []
        if len(data) < len(self):
            return []

        exact_indices = [
            index for index, mask in enumerate(self.masks) if mask == 0xFF
        ]
        matches: list[int] = []
        if exact_indices:
            anchor_index = min(
                exact_indices,
                key=lambda index: data.count(self.values[index]),
            )
            anchor = bytes((self.values[anchor_index],))
            search_from = anchor_index
            while True:
                found = data.find(anchor, search_from)
                if found < 0:
                    break
                candidate = found - anchor_index
                if self.matches(data, candidate):
                    matches.append(candidate)
                    if limit is not None and len(matches) >= limit:
                        break
                search_from = found + 1
            return matches

        for candidate in range(len(data) - len(self) + 1):
            if self.matches(data, candidate):
                matches.append(candidate)
                if limit is not None and len(matches) >= limit:
                    break
        return matches


@dataclass(frozen=True, slots=True)
class MemoryLocator:
    kind: str
    section: str = ".text"
    pattern: str | None = None
    occurrence: int = 0
    match_offset: int = 0
    rip_displacement_offset: int | None = None
    instruction_size: int | None = None
    addend: int = 0
    dereference_count: int = 0


@dataclass(frozen=True, slots=True)
class DiscoveredAddress:
    name: str
    address: int
    module_offset: int
    match_address: int | None
    details: dict[str, int | float | str]


def scan_section(
    memory: ProcessMemory,
    section: PESection,
    pattern: BytePattern,
    *,
    max_results: int,
) -> list[int]:
    if max_results <= 0:
        return []
    if len(pattern) > 1_048_576:
        raise ValueError("AOB pattern is too large")
    results: list[int] = []
    seen: set[int] = set()
    for chunk_address, chunk in memory.iter_memory(
        section.address,
        section.size,
        overlap=max(0, len(pattern) - 1),
    ):
        remaining = max_results - len(results)
        for offset in pattern.find_all(chunk, limit=remaining):
            address = chunk_address + offset
            if address not in seen:
                seen.add(address)
                results.append(address)
                if len(results) >= max_results:
                    return results
    return results


def resolve_memory_locator(
    memory: ProcessMemory,
    name: str,
    locator: MemoryLocator,
) -> DiscoveredAddress:
    if locator.kind == "elden_ring_world_chr_man":
        return EldenRingAddressDiscovery(memory).discover_world_chr_man()
    if locator.kind != "aob":
        raise AddressDiscoveryError(f"Unsupported memory locator kind: {locator.kind}")
    if locator.pattern is None:
        raise AddressDiscoveryError(f"AOB locator {name!r} has no pattern")

    pattern = BytePattern.parse(locator.pattern)
    matches = scan_section(
        memory,
        memory.section(locator.section),
        pattern,
        max_results=locator.occurrence + 1,
    )
    if len(matches) <= locator.occurrence:
        raise AddressDiscoveryError(
            f"AOB locator {name!r} found {len(matches)} matches; "
            f"occurrence {locator.occurrence} was requested"
        )
    match = matches[locator.occurrence]
    address = match + locator.match_offset
    if locator.rip_displacement_offset is not None:
        if locator.instruction_size is None:
            raise AddressDiscoveryError(
                f"AOB locator {name!r} needs instruction_size for RIP resolution"
            )
        displacement = memory.read_int32(match + locator.rip_displacement_offset)
        address = match + locator.instruction_size + displacement
    address += locator.addend
    for _ in range(locator.dereference_count):
        address = memory.read_pointer(address)
    return DiscoveredAddress(
        name=name,
        address=address,
        module_offset=address - memory.module_base,
        match_address=match,
        details={"section": locator.section},
    )


class EldenRingAddressDiscovery:
    """Discover Elden Ring globals without executing or modifying game code."""

    # Broadly identifies the common x64 singleton-null-check sequence. Candidate
    # globals are identified structurally below instead of relying on one build's
    # absolute address or calling a function inside the target process.
    _SINGLETON_NULL_CHECK = BytePattern.parse(
        "48 8B ? ? ? ? ? 48 85 ? 75 ? 48 8D 0D ? ? ? ? E8 ? ? ? ?"
    )

    WORLD_MAIN_PLAYER = 0x1E508
    WORLD_NET_PLAYERS = 0x10EF8
    CHR_MODULES = 0x190
    CHR_STATS_MODULE = 0x0
    CHR_TRANSFORM_MODULE = 0x68
    CURRENT_HP = 0x138
    MAX_HP = 0x13C
    CURRENT_FP = 0x148
    MAX_FP = 0x150
    CURRENT_STAMINA = 0x154
    MAX_STAMINA = 0x158
    POSITION = 0x70
    MAP_ID = 0x6D0

    def __init__(self, memory: ProcessMemory) -> None:
        self.memory = memory

    def discover_world_chr_man(self) -> DiscoveredAddress:
        text = self.memory.section(".text")
        data = self.memory.section(".data")
        matches = scan_section(
            self.memory,
            text,
            self._SINGLETON_NULL_CHECK,
            max_results=16_384,
        )
        candidates: list[tuple[int, int, int, dict[str, int | float | str]]] = []
        seen_globals: set[int] = set()
        for match in matches:
            try:
                instruction = self.memory.read(match, 7)
                # ModRM mod=00 and r/m=101 identifies RIP-relative memory access.
                if len(instruction) != 7 or instruction[:2] != b"\x48\x8b":
                    continue
                if (instruction[2] & 0xC7) != 0x05:
                    continue
                global_address = match + 7 + struct.unpack_from("<i", instruction, 3)[0]
                if global_address in seen_globals or not data.contains(global_address):
                    continue
                seen_globals.add(global_address)
                validated = self._validate_world_candidate(global_address)
                if validated is not None:
                    score, details = validated
                    candidates.append((score, global_address, match, details))
            except (MemoryReadError, OSError, ValueError, struct.error):
                continue

        if not candidates:
            raise AddressDiscoveryError(
                "WorldChrMan was not found. Load a character into the game world "
                "and verify that the executable layout is supported."
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score = candidates[0][0]
        best = [candidate for candidate in candidates if candidate[0] == best_score]
        if len(best) != 1:
            addresses = ", ".join(
                f"{candidate[1]:#x}"
                f"(player={candidate[3].get('player', 0):#x},"
                f"hp={candidate[3].get('player_health', '?')})"
                for candidate in best[:8]
            )
            raise AddressDiscoveryError(
                f"WorldChrMan discovery was ambiguous at score {best_score}: {addresses}"
            )
        _, address, match, details = best[0]
        return DiscoveredAddress(
            name="world_chr_man",
            address=address,
            module_offset=address - self.memory.module_base,
            match_address=match,
            details=details,
        )

    def _validate_world_candidate(
        self,
        global_address: int,
    ) -> tuple[int, dict[str, int | float | str]] | None:
        world = self.memory.read_pointer(global_address)
        score = 8
        try:
            world_vtable = self.memory.read_pointer(world)
        except (MemoryReadError, OSError):
            return None
        if self.memory.section(".rdata").contains(world_vtable):
            score += 3

        # WorldChrMan exposes the local player through both the direct field and
        # the slot-0 network-player list. Agreement is a strong discriminator,
        # but it remains a score so minor layout changes produce diagnostics.
        direct_player: int | None = None
        slot_zero_player: int | None = None
        try:
            direct_player = self.memory.read_pointer(world + self.WORLD_MAIN_PLAYER)
        except MemoryReadError:
            pass
        try:
            net_players = self.memory.read_pointer(world + self.WORLD_NET_PLAYERS)
            slot_zero_player = self.memory.read_pointer(net_players)
        except MemoryReadError:
            pass
        if direct_player is not None and direct_player == slot_zero_player:
            score += 8
        player = direct_player or slot_zero_player
        if player is None:
            return None
        chr_modules = self.memory.read_pointer(player + self.CHR_MODULES)
        stats = self.memory.read_pointer(chr_modules + self.CHR_STATS_MODULE)

        hp = self._read_i32(stats + self.CURRENT_HP)
        max_hp = self._read_i32(stats + self.MAX_HP)
        if not (0 <= hp <= 200_000 and 1 <= max_hp <= 200_000 and hp <= max_hp * 2):
            return None

        details: dict[str, int | float | str] = {
            "world": world,
            "player": player,
            "world_vtable": world_vtable,
            "main_player_matches_net_slot_zero": int(
                direct_player is not None and direct_player == slot_zero_player
            ),
            "player_health": hp,
            "player_max_health": max_hp,
        }

        module_start = self.memory.module_base
        module_end = module_start + self.memory.module_size
        vtable = self.memory.read_pointer(player)
        if module_start <= vtable < module_end:
            score += 2

        fp = self._read_i32(stats + self.CURRENT_FP)
        max_fp = self._read_i32(stats + self.MAX_FP)
        if 0 <= fp <= 100_000 and 0 <= max_fp <= 100_000 and fp <= max(1, max_fp) * 2:
            score += 2
            details.update(player_fp=fp, player_max_fp=max_fp)

        stamina = self._read_i32(stats + self.CURRENT_STAMINA)
        max_stamina = self._read_i32(stats + self.MAX_STAMINA)
        if (
            0 <= stamina <= 100_000
            and 1 <= max_stamina <= 100_000
            and stamina <= max_stamina * 2
        ):
            score += 2
            details.update(
                player_stamina=stamina,
                player_max_stamina=max_stamina,
            )

        try:
            transform = self.memory.read_pointer(chr_modules + self.CHR_TRANSFORM_MODULE)
            position = struct.unpack(
                "<fff",
                self.memory.read(transform + self.POSITION, 12),
            )
            if all(math.isfinite(value) and abs(value) < 10_000_000 for value in position):
                score += 2
                details.update(
                    player_x=float(position[0]),
                    player_y=float(position[1]),
                    player_z=float(position[2]),
                )
        except (MemoryReadError, struct.error):
            pass

        try:
            map_id = int(struct.unpack("<I", self.memory.read(player + self.MAP_ID, 4))[0])
            if map_id <= 999_999_999:
                score += 1
                details["location_id"] = map_id
        except (MemoryReadError, struct.error):
            pass
        details["validation_score"] = score
        return score, details

    def _read_i32(self, address: int) -> int:
        return int(struct.unpack("<i", self.memory.read(address, 4))[0])
