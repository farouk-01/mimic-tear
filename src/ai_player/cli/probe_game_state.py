from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from ai_player.game_state import (  # noqa: E402
    EldenRingStateReader,
    load_memory_profile,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read one state snapshot using a read-only memory profile."
    )
    parser.add_argument("profile", type=Path)
    args = parser.parse_args(argv)

    profile = load_memory_profile(args.profile)
    with EldenRingStateReader.open(profile) as reader:
        snapshot = reader.read()
    output = {
        "profile": profile.metadata(),
        "valid": snapshot.valid,
        "values": snapshot.values,
        "read_errors": snapshot.read_errors,
    }
    print(json.dumps(output, indent=2))
    if not snapshot.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
