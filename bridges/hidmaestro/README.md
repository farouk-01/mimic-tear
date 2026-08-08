# HIDMaestro controller bridge

This bridge turns AI `ControllerState` messages into a Windows-wide Xbox 360
controller that Elden Ring can read. HIDMaestro v1.5.0 is downloaded from its
official GitHub release and verified against the pinned SHA-256 in
`bootstrap.ps1`.

## Setup

From PowerShell at the repository root:

```powershell
.\bridges\hidmaestro\bootstrap.ps1
.\bridges\hidmaestro\start.ps1
```

Approve the Windows administrator prompt. The elevated bridge installs
HIDMaestro idempotently, creates an Xbox 360 controller, and waits on the local
named pipe `\\.\pipe\ai-player-controller`.

The controller releases all input if the AI disconnects or fails to submit a
state for 250 ms. Closing the bridge removes the virtual controller.

With the bridge waiting for a client, run the current policy from the
repository root:

```powershell
.\play.cmd .\artifacts\policy-spatial-expanded-20260805\best.pt --armed
```

Run the neutral end-to-end probe from an elevated terminal with:

```powershell
.\bridges\hidmaestro\bin\acl-fixed\ai-player-controller-bridge.exe probe
```

The SDL3 controller remains useful for in-process tests. This HIDMaestro bridge
is the output path used for a separate game process.
