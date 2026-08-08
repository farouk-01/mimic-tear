# Recording and replay

The recorder writes synchronized game frames and controller state into an
atomically finalized session directory. New recordings default to 640x360 at
30 FPS; training can still resize those frames to 320x180.

## Record

From the project root:

```powershell
.\.venv\Scripts\python.exe -m ai_player.cli.record `
  --theme exploration `
  --tag basic-movement `
  --split train `
  --label movement
```

`--theme` is the broad category and `--tag` is the specific recording tag/class.
Themes can be nested with forward slashes, for example `movement/jump`.
The command above saves to `recordings/train/exploration/basic-movement/` using
the fixed layout `recordings/<split>/<theme>/<tag>/`. All path components must
use lowercase letters, numbers, hyphens, or underscores. A destination is never
overwritten, so choose a new tag such as `basic-movement-02` when recording
that class again.

`--split` assigns the whole session to either `train` or `validation`; frames
from one session are never divided across both sets. Use repeatable `--label`
flags for additional reusable, manually supplied labels. Record validation
sessions separately, ideally on a different day from the training sessions.
The recorder registers a global `F8` stop hotkey by default, so it can stop
even when the preview window is not focused. Use `--stop-hotkey CTRL+F8` to
choose another function/navigation key combination, or `--stop-hotkey NONE`
to disable global registration. `CTRL+F9` cancels by default: it discards the
in-progress session instead of finalizing it. Configure it with
`--cancel-hotkey`, or use `--cancel-hotkey NONE` to disable it.

The preview gives a three-second countdown and shows measured FPS, estimated
drops, input synchronization offset, free disk space, and live controller
state. Press Q or Esc in the preview, or Ctrl+C in the terminal, to finish the
session. Once the capture validates and is finalized, its replay window opens
automatically.

Useful options:

```powershell
# Capture a specific monitor at the defaults.
.\.venv\Scripts\python.exe -m ai_player.cli.record --theme combat --tag boss-practice --split train --monitor 1

# Capture a desktop region expressed as left,top,right,bottom.
.\.venv\Scripts\python.exe -m ai_player.cli.record --theme exploration --tag interior-01 --split train --region 100,80,1820,1040

# Limit a headless collection run to ten minutes.
.\.venv\Scripts\python.exe -m ai_player.cli.record --theme combat --tag group-01 --split train --no-preview --max-duration 600

# Cancel and discard an in-progress recording with CTRL+F9.
.\.venv\Scripts\python.exe -m ai_player.cli.record --theme movement --tag basic-movement --split train --cancel-hotkey CTRL+F9

# Also write a human-readable CSV mirror.
.\.venv\Scripts\python.exe -m ai_player.cli.record --theme items --tag healing-01 --split validation --csv
```

Run `record.py --help` for resolution, FPS, codec, deadzone, countdown, and
disk-space controls.

## Repeated boss-attempt recording

The opt-in boss loop automates the practice-tool sequence after every attempt:
it finalizes the recording, sends `P`, waits for gameplay to unload, sends
`Ctrl+O` to load the currently selected savefile, sends `E` to Continue,
waits for gameplay to become readable again, and starts the next recording.
Each episode gets a unique suffix such as `soldier-of-godrick-0001`. Replays do
not open between episodes.

Before starting, load the character at the fog gate, select the desired
savefile in the practice tool, hide the overlay, and leave Elden Ring focused.
The practice tool must retain its default `P` quitout and `Ctrl+O` load-savefile
bindings. Run offline without Easy Anti-Cheat, stop the virtual-controller
bridge, and connect the physical controller you want recorded.

For the Soldier of Godrick setup, run:

```powershell
.\record-soldier-loop.cmd
```

Play the attempt, then press `F10`. The recorder saves the episode and performs
the reset before its normal three-second countdown starts the next attempt.
`Ctrl+F9` discards the current episode and exits the loop; `Ctrl+C` in the
terminal also stops it. F10 is used because the practice tool binds F8 to a
hitbox option.

The wrapper accepts extra recorder flags. For example, stop after ten saved
episodes or capture monitor 1:

```powershell
.\record-soldier-loop.cmd --boss-episodes 10 --monitor 1
```

To use the loop for another encounter, invoke `python -m ai_player.cli.record`
with your own `--theme`, base `--tag`, and `--split`, plus `--boss-loop` and
`--no-preview`. Timing can be adjusted with `--boss-title-settle`,
`--boss-snapshot-delay`, `--boss-gameplay-settle`, and
`--boss-reset-timeout` if the machine loads unusually slowly.

Controller state is sampled independently at 250 Hz by default. Each video
frame is paired with the closest timestamped controller sample. Use
`--input-hz` to change the polling rate and `--max-sync-offset-ms` to configure
the validation limit (15 ms by default). The controller sampler uses a
high-resolution wait near each polling deadline to avoid stale samples caused
by coarse Windows timer waits. If a frame still has no controller sample
within that limit, the frame is dropped instead of being written with a stale
action label; the recorder reports the synchronization-drop count when it
finishes.

## Session safety

Sessions are first written to a hidden `.<tag>.<timestamp>.inprogress`
directory inside the theme directory. The recorder closes both files, fully
decodes the video, validates every input value, and only then renames the
directory to `<name>`. If capture or validation fails, the in-progress directory
and its `metadata.json` are retained for diagnosis and ignored by dataset
discovery.

Each completed session contains:

- `frames.mp4`: clean model input without the preview HUD.
- `inputs.parquet`: canonical typed, compressed controller inputs with frame
  indices, timestamps, synchronization offsets, analog axes, and buttons.
- `inputs.csv`: optional human-readable mirror created only with `--csv`.
- `game_state.parquet`: frame-aligned, read-only process-memory state. It is
  enabled by default and can be disabled explicitly with `--no-game-state`.
- `metadata.json`: classification and dataset split, capture configuration,
  exact source/output preprocessing, controller identity, timing/drop and sync
  telemetry, per-input active ratios, button press counts and rates, analog
  summaries, and stop reason.

Training requires at least one complete session assigned to `train` and one
assigned to `validation`. The loader keeps each recording entirely within its
declared split.

## Training frame cache

Training and evaluation automatically decode each MP4 once, in sequential
order, into a memory-mapped BGR `uint8` array at the configured model input
resolution. Caches are stored inside each session under `.frame_cache/`, and
array row `i` always corresponds to video and Parquet `frame_index == i`.
Subsequent epochs perform direct indexed reads instead of seeking through the
compressed MP4.

The cache is rebuilt automatically when the source video's size or modification
time, reported frame count, or target resolution changes. Use
`--rebuild-frame-cache` to force a rebuild, or `--no-frame-cache` to use direct
MP4 decoding. The cache is uncompressed and requires approximately
`frames * width * height * 3` bytes of disk space. Data loading defaults to up
to four worker processes and uses persistent workers and prefetching.

## Frame-aligned read-only Elden Ring state

The recorder can sample a configured `eldenring.exe` process through the
Windows `OpenProcess` and `ReadProcessMemory` APIs. It requests only process
query and memory-read permissions; there is no memory writing, code injection,
hooking, or anti-cheat bypass functionality. Use this only in a controlled
offline environment where attaching a diagnostic reader is permitted. Do not
attach it while anti-cheat protection is active.

The discovery tool and recorder refuse to attach when an Easy Anti-Cheat
process is detected. Launcher process names, including
`eldenring.exe`, do not trigger this guard. The guard is checked
again once per second during capture; if Easy Anti-Cheat appears after
attachment, game-state sampling fails and the recording is stopped without
being finalized.

The bundled [Elden Ring profile](../configs/game-state/elden-ring.json) discovers the
`WorldChrMan` global from the running executable's instruction layout, then
validates the candidate using plausible player pointers and health values. It
does not depend on a permanent module offset. Relative structure offsets are
kept in the profile so they remain visible and testable.

Load a character into the game world, then test discovery by itself:

```powershell
.\.venv\Scripts\python.exe -m ai_player.cli.discover_game_state
```

Test a filled profile with a single read before recording:

```powershell
.\.venv\Scripts\python.exe -m ai_player.cli.probe_game_state `
  configs\game-state\elden-ring.json
```

The bundled profile is used automatically by the record command and the boss
loop. Pass `--no-game-state` only for diagnostics that intentionally should not
produce state-aware training data.

A field may start from either `module_base + base_offset` or a discovered
`base_locator`. For every pointer offset, resolution performs `address =
*(address) + offset`; the configured type is read from the final address.
String fields require a fixed maximum `length`. A field with `scope: "static"`
is read once when it first becomes available, then its cached value is repeated
into every aligned frame row. Dynamic fields are read on every sampler poll.

The currently resolved state schema includes:

- Player health, FP, stamina, and position.
- Player lock-on state.
- Location ID.

Unresolved fields are documented in `src/ai_player/game_state/stubs.py` and are
not written to Parquet or passed to the model. They currently include enemy
identity/health/position, camera orientation, flasks, quick items and counts,
current hand/weapon/two-hand state, spells, level, attributes, weapon upgrade
levels, armor, and talismans. This prevents guessed or version-stale offsets
from silently becoming training data.

A failed required field makes a snapshot invalid. The first snapshot must be
valid or recording aborts; later invalid samples are retained with
`state_valid=false` and availability masks in the model vector. Every state row
has the same frame index, timestamp, and frame timestamp as its controller row.

Use `--game-state-hz` to change the 60 Hz default and
`--max-game-state-sync-offset-ms` to change the 25 ms validation limit.

Training now requires `game_state.parquet` for every selected session. The
policy encodes normalized HP/FP/stamina ratios, lock-on state, scaled position,
and per-value availability masks with a state MLP, then fuses those features
with the CNN image representation before producing controller outputs.

## Validate

```powershell
.\.venv\Scripts\python.exe -m ai_player.cli.validate recordings\train\exploration\basic-movement
```

Add `--json` for machine-readable output. The recorder supports one strict,
typed Parquet format; recordings with different columns or types are rejected.

## Replay and exclude bad frames

```powershell
.\.venv\Scripts\python.exe -m ai_player.cli.replay recordings\train\exploration\basic-movement
```

The replay window shows the exact Parquet controller row paired with each video
frame, including stick positions, trigger pressure, active buttons, and input
synchronization offset. To overlay the AI's HiResCAM evidence on a recording,
supply the policy checkpoint that you want to inspect:

```powershell
.\.venv\Scripts\python.exe -m ai_player.cli.replay recordings\train\exploration\basic-movement --cam-checkpoint artifacts\cnn-soldier-godrick-1\best.pt
```

Press `H` to toggle HiResCAM during replay. `--cam-fps` controls how often the
map is recomputed while playing and `--cam-opacity` controls the blend.

- Space plays or pauses.
- Left/right or A/D moves one frame; J/L moves one second.
- Home/end or G/E jumps to the beginning or end.
- `[` starts a selection and `]` ends it.
- X or Delete excludes the selected range from training.
- U restores the most recently excluded range; C clears the selection.
- S saves; Q exits and automatically saves pending changes.

Exclusions are stored in `annotations.json`. The source MP4 and Parquet files
are never modified, and the dataset loader automatically skips excluded frames.
