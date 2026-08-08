# Project architecture

All Python production code lives under `mimic_tear`. The three top-level
Windows command files are intentionally thin launchers for the canonical
scripts and installed commands.

```text
mimic-tear/
|-- pyproject.toml
|-- configs/
|   `-- game-state/          Versioned memory profiles
|-- mimic_tear/
|   |-- cli/                 Argument parsing and command entry points
|   |-- controller/          Controller state and bridge client
|   |-- dataset/             Session discovery, decoding, transforms, loaders
|   |-- game_state/          Elden Ring state schema, resolver, reader, features
|   |-- platform/windows/    Windows process-memory and hotkey primitives
|   |-- policy/              Neural network, checkpoints, and policy loss
|   |-- recording/           Capture, alignment, persistence, replay, validation
|   |-- runtime/             Live inference orchestration
|   |-- training/            Training orchestration
|   |-- evaluation/          Offline metrics and checkpoint evaluation
|   `-- visualization/       HiResCAM and controller overlays
|-- bridges/hidmaestro/      Elevated system-controller bridge
|-- native/                  Native controller extension
|-- examples/                Standalone diagnostics and examples
|-- scripts/windows/         Windows workflow wrappers
|-- tests/                   Unit and integration tests by subsystem
|-- recordings/              Generated datasets (ignored)
`-- artifacts/               Generated checkpoints/reports (ignored)
```

## Dependency direction

The low-level layers (`platform`, schemas, and controller state) do not import
orchestration code. Domain packages (`game_state`, `dataset`, `policy`) can use
those primitives. `recording` and `runtime` compose the domains, while `cli`
only exposes commands. Visualization consumes policy output but policy code
never depends on visualization.

## Adding functionality

- New memory fields belong in `game_state`; OS memory APIs belong in
  `platform/windows`.
- Capture synchronization and persisted session formats belong in `recording`.
- Model inputs, architecture, checkpoint compatibility, and objectives belong
  in `policy`.
- On-screen debugging belongs in `visualization`, never in the policy model.
- Commands should be small adapters in `cli`, with behavior implemented in a
  domain or orchestration package.
