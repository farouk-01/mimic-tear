# Mimic Tear quick start

Install the package in the project virtual environment once so the documented
module and console entry points resolve consistently:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run commands from the project root in PowerShell:

```powershell
cd C:\Users\farou\Documents\projects\mimic-tear
.\.venv\Scripts\Activate.ps1
```

Use Elden Ring offline with Easy Anti-Cheat disabled. The practice tool should
be loaded only for the recording reset workflow; hide its overlay with
Right Shift + 0 before recording or live inference.

## Record repeated Soldier of Godrick attempts

Before starting:

1. Connect the physical Xbox controller that you want recorded.
2. Stop the HIDMaestro controller bridge if it is running. This prevents the
   recorder from selecting the virtual controller instead of the physical one.
3. Load the character at the fog gate.
4. Select the pre-fight savefile in the practice tool.
5. Keep the practice-tool bindings `P` for quitout and `Ctrl+O` for loading the
   selected savefile.
6. Hide the practice-tool overlay and focus Elden Ring.

Start a training recording loop:

```powershell
.\record-soldier-loop.cmd
```

The current attempt begins recording after a three-second countdown. Press
`F10` when the attempt is finished. The loop then:

1. Finalizes the current recording.
2. Presses `P` and waits for gameplay to unload.
3. Presses `Ctrl+O` to restore the selected savefile.
4. Waits three seconds.
5. Presses the Elden Ring confirm key three times, 1.5 seconds apart.
6. Waits until gameplay is readable, then starts the next countdown.

Episodes receive unique names such as `soldier-of-godrick-0001` and continue
from the next available number after restarting the script. Training sessions
are stored under:

```text
recordings\train\gameplay\boss\soldier-of-godrick\
```

Record validation attempts with the same loop:

```powershell
.\record-soldier-loop.cmd --split validation
```

Validation numbering is independent and its recordings are stored under
`recordings\validation\gameplay\boss\soldier-of-godrick\`.

Useful variations:

```powershell
# Stop after 10 finalized attempts.
.\record-soldier-loop.cmd --boss-episodes 10

# Capture the second DXGI monitor instead of monitor 0.
.\record-soldier-loop.cmd --monitor 1

# Record 10 validation attempts from monitor 1.
.\record-soldier-loop.cmd --split validation --boss-episodes 10 --monitor 1
```

- `F10`: save the current attempt and perform the reset.
- `Ctrl+F9`: discard the in-progress attempt and stop the loop.
- `Ctrl+C`: stop from the terminal.

The recorder captures only the selected monitor and does not record desktop
audio. A video on another monitor will not appear in the dataset, although it
can increase GPU load and cause dropped frames.

## Let the AI play through the virtual controller

The AI sends controller state to an elevated HIDMaestro bridge over the local
named pipe `\\.\pipe\mimic-tear-controller`. The bridge creates a system-wide
virtual Xbox 360 controller that Elden Ring can read.

### One-time bridge setup

If the bridge has not been built on this checkout, run:

```powershell
.\bridges\hidmaestro\bootstrap.ps1
```

This downloads the pinned HIDMaestro release, verifies its SHA-256, and builds
the bridge. Approve installation/elevation prompts when shown.

### Start a live AI session

For the most reliable controller assignment, disconnect the physical Xbox
controller before starting the virtual bridge. Then:

1. Start the bridge:

   ```powershell
   .\bridges\hidmaestro\start.ps1
   ```

2. Approve the Windows administrator prompt and wait for:

   ```text
   System controller ready: Xbox 360 Controller (Wired)
   Waiting for AI client...
   ```

   `Wired` is the virtual HIDMaestro profile name; it does not describe the
   physical controller or its dongle.

3. Load Elden Ring into the game world, hide the practice tool, and keep the
   game visible on the captured monitor.

4. In the normal project PowerShell window, run the latest Soldier policy:

   ```powershell
   .\play.cmd .\artifacts\cnn-soldier-godrick-1\best.pt --armed --debug
   ```

   `--armed` is mandatory because this command sends live input. `--debug`
   prints the AI's controller outputs and synchronized memory state once per
   second and can be omitted. State-aware checkpoints automatically use the
   bundled read-only memory profile at 60 Hz. Each visual decision is paired
   with the nearest state sample and live play stops if their timestamps drift
   by more than 25 ms. Override these settings with `--game-state-profile`,
   `--game-state-hz`, and `--max-game-state-sync-offset-ms`.

   Add `--controller-overlay` to display a click-through controller HUD over
   Elden Ring. The HUD uses Windows `WDA_EXCLUDEFROMCAPTURE`, so it remains
   visible on the monitor but is omitted from the AI's DXGI frames:

   ```powershell
   .\play.cmd .\artifacts\cnn-soldier-godrick-1\best.pt --armed --controller-overlay
   ```

   The overlay requires borderless/windowed presentation so Windows can
   compose it above the game. Startup fails before controller output is armed
   if Windows cannot confirm the capture-exclusion flag.

   Add `--cam-overlay` to visualize the screen regions supporting the AI's
   current controller action with HiResCAM:

   ```powershell
   .\play.cmd .\artifacts\cnn-soldier-godrick-1\best.pt --armed --cam-overlay
   ```

   Press `F8` at any time to hide or show the CAM. The heatmap is computed in
   a separate process (5 updates/second by default), is click-through, and is
   excluded from DXGI capture so it cannot feed back into the policy. Tune it
   with `--cam-fps`, `--cam-opacity`, and `--cam-threshold`. This uses the
   `HiResCAM` algorithm from the `grad-cam` Python package, not GradCAM.

5. If Elden Ring is on the second DXGI output, select it explicitly:

   ```powershell
   .\play.cmd .\artifacts\cnn-soldier-godrick-1\best.pt --armed --output-index 1 --debug
   ```

6. Press `Ctrl+C` in the AI terminal to stop inference and release all buttons.
   Then press `Ctrl+C` in the bridge window to remove the virtual controller.

To run another checkpoint, replace the `.pt` path in the command. The bridge
must display `AI client connected` after `play.cmd` starts. If the AI reports
pipe access denied or cannot connect, close the old bridge with `Ctrl+C`, run
`start.ps1` again from the same Windows account, approve UAC, and retry.

## Safety notes

- Use the process-memory reader and practice tool only in a permitted offline
  environment with anti-cheat inactive.
- Keep the terminal available so `Ctrl+C` can immediately stop AI output.
- The bridge watchdog neutralizes the virtual controller if the AI stops
  sending state for 250 ms.
- Do not leave the bridge running while recording physical-controller
  demonstrations; stop it first so the correct controller is sampled.

More details are available in [the recording guide](docs/recording.md),
[architecture guide](docs/architecture.md), and
[controller bridge guide](bridges/hidmaestro/README.md).
