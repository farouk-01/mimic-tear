from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from ai_player.game_state.discovery import (  # noqa: E402
    BytePattern,
    EldenRingAddressDiscovery,
)
from ai_player.platform.windows.process_memory import (  # noqa: E402
    AntiCheatDetectedError,
    MemoryReadError,
    PESection,
    ProcessMemory,
    RunningProcess,
    assert_anti_cheat_inactive,
    is_anti_cheat_process_name,
)
from ai_player.game_state.profile import (  # noqa: E402
    EldenRingMemoryProfile,
    MemoryField,
    load_memory_profile,
)
from ai_player.game_state.reader import (  # noqa: E402
    EldenRingStateReader,
    GameStateSnapshot,
)
from ai_player.game_state.schema import GAME_STATE_PARQUET_SCHEMA  # noqa: E402
from ai_player.recording.game_state_capture import (  # noqa: E402
    GameStateSample,
    GameStateSampler,
    GameStateWriter,
)
from ai_player.recording.record import parse_args  # noqa: E402
from tests.recording.test_recording import (  # noqa: E402
    current_row,
    write_inputs,
    write_video,
)
from ai_player.recording.validation import validate_session  # noqa: E402


class FakeMemory:
    def __init__(self) -> None:
        self.closed = False
        self.read_count = 0

    def resolve(self, base_offset: int, pointer_offsets: tuple[int, ...]) -> int:
        return base_offset + sum(pointer_offsets)

    def read_typed(
        self,
        address: int,
        value_type: str,
        *,
        length: int | None,
    ) -> object:
        del value_type, length
        self.read_count += 1
        values = {0x10: 500, 0x30: "Gatefront"}
        if address not in values:
            raise MemoryReadError(f"unreadable test address {address:#x}")
        return values[address]

    def resolve_address(
        self,
        base_address: int,
        pointer_offsets: tuple[int, ...],
    ) -> int:
        return base_address + sum(pointer_offsets)

    def close(self) -> None:
        self.closed = True


class SyntheticEldenRingMemory:
    def __init__(self) -> None:
        self.module_base = 0x140000000
        self.module_size = 0x5000
        self._text = PESection(".text", self.module_base + 0x1000, 0x100)
        self._data = PESection(".data", self.module_base + 0x3000, 0x100)
        self._rdata = PESection(".rdata", self.module_base + 0x4000, 0x100)
        self._memory: dict[int, int] = {}

        match = self._text.address + 0x20
        global_address = self._data.address + 0x18
        displacement = global_address - (match + 7)
        instruction = (
            b"\x48\x8B\x05"
            + struct.pack("<i", displacement)
            + b"\x48\x85\xC0\x75\x2E"
            + b"\x48\x8D\x0D\x11\x22\x33\x44"
            + b"\xE8\x55\x66\x77\x00"
        )
        self._write(self._text.address, b"\x90" * self._text.size)
        self._write(match, instruction)

        world = 0x50000000
        player = 0x51000000
        net_players = 0x51500000
        chr_modules = 0x52000000
        stats = 0x53000000
        transform = 0x54000000
        self._write_pointer(global_address, world)
        self._write_pointer(world, self._rdata.address + 0x20)
        self._write_pointer(
            world + EldenRingAddressDiscovery.WORLD_MAIN_PLAYER,
            player,
        )
        self._write_pointer(world + EldenRingAddressDiscovery.WORLD_NET_PLAYERS, net_players)
        self._write_pointer(net_players, player)
        self._write_pointer(player, self.module_base + 0x2200)
        self._write_pointer(player + EldenRingAddressDiscovery.CHR_MODULES, chr_modules)
        self._write_pointer(
            chr_modules + EldenRingAddressDiscovery.CHR_STATS_MODULE,
            stats,
        )
        self._write_pointer(
            chr_modules + EldenRingAddressDiscovery.CHR_TRANSFORM_MODULE,
            transform,
        )
        self._write_i32(stats + EldenRingAddressDiscovery.CURRENT_HP, 500)
        self._write_i32(stats + EldenRingAddressDiscovery.MAX_HP, 600)
        self._write_i32(stats + EldenRingAddressDiscovery.CURRENT_FP, 80)
        self._write_i32(stats + EldenRingAddressDiscovery.MAX_FP, 100)
        self._write_i32(stats + EldenRingAddressDiscovery.CURRENT_STAMINA, 90)
        self._write_i32(stats + EldenRingAddressDiscovery.MAX_STAMINA, 120)
        self._write(
            transform + EldenRingAddressDiscovery.POSITION,
            struct.pack("<fff", 1.0, 2.0, 3.0),
        )
        self._write_i32(player + EldenRingAddressDiscovery.MAP_ID, 12345)

    def _write(self, address: int, value: bytes) -> None:
        self._memory.update(
            {address + index: byte for index, byte in enumerate(value)}
        )

    def _write_pointer(self, address: int, value: int) -> None:
        self._write(address, struct.pack("<Q", value))

    def _write_i32(self, address: int, value: int) -> None:
        self._write(address, struct.pack("<i", value))

    def read(self, address: int, size: int) -> bytes:
        try:
            return bytes(self._memory[address + index] for index in range(size))
        except KeyError as error:
            raise MemoryReadError(f"unreadable synthetic address {address:#x}") from error

    def read_pointer(self, address: int) -> int:
        value = struct.unpack("<Q", self.read(address, 8))[0]
        if not value:
            raise MemoryReadError(f"null synthetic pointer {address:#x}")
        return int(value)

    def read_int32(self, address: int) -> int:
        return int(struct.unpack("<i", self.read(address, 4))[0])

    def section(self, name: str) -> PESection:
        return {".text": self._text, ".data": self._data, ".rdata": self._rdata}[name]

    def iter_memory(
        self,
        address: int,
        size: int,
        *,
        overlap: int = 0,
        chunk_size: int = 1_048_576,
    ):
        del overlap, chunk_size
        yield address, self.read(address, size)


class FakeReader:
    def __init__(self, snapshot: GameStateSnapshot) -> None:
        self.snapshot = snapshot
        self.closed = False

    def read(self) -> GameStateSnapshot:
        return self.snapshot

    def close(self) -> None:
        self.closed = True


def test_profile(path: Path) -> EldenRingMemoryProfile:
    return EldenRingMemoryProfile(
        name="test-profile",
        game_version="test-build",
        process_name="eldenring.exe",
        module_name="eldenring.exe",
        pointer_size=8,
        fields={
            "player_health": MemoryField(0x10, (), "int32", required=True),
            "player_stamina": MemoryField(0x20, (), "int32"),
            "location_id": MemoryField(0x30, (), "int32"),
        },
        source_path=path,
    )


class GameStateTests(unittest.TestCase):
    def test_anti_cheat_guard_blocks_only_easy_anti_cheat_processes(self) -> None:
        self.assertTrue(is_anti_cheat_process_name("EasyAntiCheat_EOS.exe"))
        # Some local launch configurations expose the game under this name
        # even when Easy Anti-Cheat is not running; only actual EAC processes
        # should block the read-only sampler.
        self.assertFalse(is_anti_cheat_process_name("eldenring.exe"))
        self.assertFalse(is_anti_cheat_process_name("eac_launcher.exe"))
        self.assertFalse(is_anti_cheat_process_name("eldenring.exe"))
        with patch(
            "ai_player.platform.windows.process_memory.find_anti_cheat_processes",
            return_value=(RunningProcess(123, "EasyAntiCheat_EOS.exe"),),
        ):
            with self.assertRaisesRegex(
                AntiCheatDetectedError,
                r"EasyAntiCheat_EOS\.exe \(PID 123\)",
            ):
                assert_anti_cheat_inactive()

    @unittest.skipUnless(os.name == "nt", "ReadProcessMemory is Windows-only")
    def test_read_process_memory_backend_is_read_only_and_operational(self) -> None:
        executable_name = Path(sys.executable).name
        with ProcessMemory.open_process_id(
            os.getpid(),
            module_name=executable_name,
        ) as memory:
            self.assertEqual(memory.read(memory.module_base, 2), b"MZ")
            self.assertIn(".text", {section.name for section in memory.pe_sections()})

    def test_aob_parser_supports_byte_and_nibble_wildcards(self) -> None:
        pattern = BytePattern.parse("48 8B ?5 ?? 4?")
        self.assertEqual(pattern.find_all(b"\x48\x8b\x35\xff\x4a"), [0])
        self.assertFalse(pattern.matches(b"\x48\x8b\x34\xff\x5a", 0))

    def test_world_chr_man_is_discovered_and_structurally_validated(self) -> None:
        memory = SyntheticEldenRingMemory()
        result = EldenRingAddressDiscovery(memory).discover_world_chr_man()  # type: ignore[arg-type]
        self.assertEqual(result.module_offset, 0x3018)
        self.assertEqual(result.details["player_health"], 500)
        self.assertEqual(result.details["location_id"], 12345)

    def test_profile_loads_hex_offsets_and_rejects_wrong_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "test",
                        "game_version": "1.0",
                        "process_name": "eldenring.exe",
                        "module_name": "eldenring.exe",
                        "pointer_size": 8,
                        "fields": {
                            "player_health": {
                                "base_locator": "world_chr_man",
                                "pointer_offsets": ["0x10", 32],
                                "type": "int32",
                                "required": True,
                            },
                            "location_id": {
                                "base_offset": "0x5678",
                                "type": "uint32",
                            },
                        },
                        "locators": {
                            "world_chr_man": {
                                "kind": "elden_ring_world_chr_man"
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            profile = load_memory_profile(profile_path)
            self.assertEqual(len(profile.sha256), 64)
            self.assertIsNone(profile.fields["player_health"].base_offset)
            self.assertEqual(
                profile.fields["player_health"].base_locator,
                "world_chr_man",
            )
            self.assertEqual(
                profile.fields["player_health"].pointer_offsets,
                (0x10, 32),
            )

            invalid = json.loads(profile_path.read_text(encoding="utf-8"))
            invalid["fields"]["player_health"]["type"] = "utf8"
            profile_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected a int type"):
                load_memory_profile(profile_path)

    def test_reader_keeps_optional_lock_on_fields_nullable(self) -> None:
        memory = FakeMemory()
        reader = EldenRingStateReader(test_profile(Path("test.json")), memory)
        snapshot = reader.read()
        self.assertTrue(snapshot.valid)
        self.assertEqual(snapshot.values["player_health"], 500)
        self.assertIsNone(snapshot.values["player_stamina"])
        self.assertIsNone(snapshot.values["location_id"])
        self.assertTrue(any("player_stamina" in error for error in snapshot.read_errors))
        reader.close()
        self.assertTrue(memory.closed)

    def test_static_profile_fields_are_read_once_and_reused(self) -> None:
        memory = FakeMemory()
        profile = EldenRingMemoryProfile(
            name="static-test",
            game_version="test",
            process_name="eldenring.exe",
            module_name="eldenring.exe",
            pointer_size=8,
            fields={
                "player_health": MemoryField(
                    0x10,
                    (),
                    "int32",
                    required=True,
                    scope="static",
                )
            },
            source_path=Path("test.json"),
        )
        reader = EldenRingStateReader(profile, memory)
        first = reader.read()
        second = reader.read()
        self.assertEqual(first.values["player_health"], 500)
        self.assertEqual(second.values["player_health"], 500)
        self.assertEqual(memory.read_count, 1)

    def test_sampler_and_writer_preserve_frame_alignment(self) -> None:
        snapshot = GameStateSnapshot(
            values={
                "player_health": 500,
                "lock_on_active": True,
                "enemy_id": 42,
                "enemy_health": 120,
                "location_id": 1001,
                "location_name": "Gatefront",
            },
            valid=True,
            read_errors=(),
        )
        reader = FakeReader(snapshot)
        sampler = GameStateSampler(lambda: reader, polling_hz=60)
        sampler.start()
        try:
            target_ns = time.perf_counter_ns()
            sample = sampler.closest(target_ns, timeout_seconds=0.05)
            self.assertTrue(sample.snapshot.valid)
            self.assertGreaterEqual(sampler.stats().valid_sample_count, 1)
        finally:
            sampler.stop()
        self.assertTrue(reader.closed)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "game_state.parquet"
            with GameStateWriter(path) as writer:
                writer.write(
                    frame_index=0,
                    timestamp_ns=target_ns,
                    frame_timestamp_ns=target_ns,
                    sample=sample,
                )
            table = pq.read_table(path)
            self.assertTrue(
                table.schema.equals(
                    GAME_STATE_PARQUET_SCHEMA,
                    check_metadata=False,
                )
            )
            row = table.to_pylist()[0]
            self.assertEqual(row["frame_index"], 0)
            self.assertEqual(row["player_health"], 500)
            self.assertTrue(row["lock_on_active"])

    def test_validator_checks_game_state_row_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "combat" / "lock-on-01"
            session.mkdir(parents=True)
            write_video(session / "frames.mp4", frame_count=1)
            write_inputs(session / "inputs.parquet", [current_row(0)])
            frame_timestamp_ns = int(current_row(0)["frame_timestamp_ns"])
            sample = GameStateSample(
                timestamp_ns=frame_timestamp_ns + 1_000_000,
                snapshot=GameStateSnapshot(
                    values={"player_health": 500, "lock_on_active": False},
                    valid=True,
                    read_errors=(),
                ),
            )
            with GameStateWriter(session / "game_state.parquet") as writer:
                writer.write(
                    frame_index=0,
                    timestamp_ns=int(current_row(0)["timestamp_ns"]),
                    frame_timestamp_ns=frame_timestamp_ns,
                    sample=sample,
                )
            (session / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "config": {
                            "theme": "combat",
                            "tag": "lock-on-01",
                            "split": "train",
                            "labels": ["lock-on"],
                            "maximum_game_state_sync_offset_ms": 25.0,
                        },
                        "files": {
                            "video": "frames.mp4",
                            "inputs": "inputs.parquet",
                            "game_state": "game_state.parquet",
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = validate_session(session)
            self.assertTrue(report.valid, report.errors)
            self.assertEqual(report.game_state_rows, 1)

    def test_game_state_profile_is_enabled_by_default_and_can_be_disabled(self) -> None:
        required = [
            "--theme",
            "exploration",
            "--tag",
            "movement-01",
            "--split",
            "train",
        ]
        self.assertEqual(
            parse_args(required).game_state_profile.name,
            "elden-ring.json",
        )
        self.assertIsNone(parse_args([*required, "--no-game-state"]).game_state_profile)
        configured = parse_args(
            [*required, "--game-state-profile", "profile.json"]
        )
        self.assertEqual(configured.game_state_profile.name, "profile.json")


if __name__ == "__main__":
    unittest.main()
