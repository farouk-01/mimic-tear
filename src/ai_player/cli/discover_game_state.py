from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from ai_player.game_state import EldenRingAddressDiscovery  # noqa: E402
from ai_player.platform.windows.process_memory import ProcessMemory  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and validate Elden Ring's WorldChrMan address using only "
            "OpenProcess and ReadProcessMemory."
        )
    )
    parser.add_argument("--process", default="eldenring.exe")
    parser.add_argument("--module", default="eldenring.exe")
    args = parser.parse_args(argv)

    with ProcessMemory.open(
        args.process,
        module_name=args.module,
        anti_cheat_guard=True,
    ) as memory:
        result = EldenRingAddressDiscovery(memory).discover_world_chr_man()
        output = {
            "process_id": memory.process_id,
            "module": memory.module_name,
            "module_base": f"0x{memory.module_base:X}",
            "name": result.name,
            "address": f"0x{result.address:X}",
            "module_offset": f"0x{result.module_offset:X}",
            "pattern_match": (
                f"0x{result.match_address:X}"
                if result.match_address is not None
                else None
            ),
            "validation": result.details,
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
