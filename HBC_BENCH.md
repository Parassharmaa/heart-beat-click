# HBC-Bench — Benchmark for Duplex-Necessary GUI Tasks

Date: 2026-05-14. Goal: measure capabilities that screenshot-loop agents are *structurally* blind to, not just bad at. If a discrete `click(x,y)` + `wait(t)` API can solve it given a strong enough planner, the task does not belong here.

## Inclusion test

A task belongs in HBC-Bench iff at least one of these holds:

1. **Trajectory-bound**: success depends on the cursor *path*, not just endpoints (signature drawing, freehand selection, gesture, drag-to-resize with snap-target preview).
2. **Mid-action observation**: success depends on observing UI change *while acting* (drag-and-drop with autoscroll, slider with live preview, color picker, video timeline scrub).
3. **Hard timing budget**: success requires action at a specific frame (catching a falling element, dismissing a modal before timeout, dragging at the right tempo).
4. **Continuous feedback control**: target moves or autosnaps and the agent must close the loop (zoom-to-fit on a moving target, balancing a slider against a metric, drag-to-reorder with auto-scroll).
5. **Multi-stream coordination**: simultaneous audio + visual + action (mute on detected siren, push-to-talk during screen share, voice-controlled UI with visual confirmation).

A task does **not** belong if it can be reduced to a sequence of static screenshots plus discrete tool calls. Filing an expense report, booking a flight, filling a form — those go to OSWorld / WebArena.

## Initial task families (v0)

| Family | Example task | Why duplex-necessary |
|---|---|---|
| **Signature & sketch** | Sign a PDF in DocuSign matching a target glyph; draw a logo in Figma free-hand to within IoU > 0.8 of target | Trajectory-bound (1) |
| **Drag-with-autoscroll** | Move a Trello card from top of a long list to a buried position in another list; live drag, scroll auto-engages | Mid-action observation (2) + continuous feedback (4) |
| **Slider live preview** | Set photo exposure in Lightroom to make histogram cross a threshold | Continuous feedback (4) |
| **Video scrub** | Scrub a YouTube video to the moment a specific event occurs (cap = 5s real time); set bookmarks at 3 specific scenes in a 90s clip | Mid-action observation (2) + timing (3) |
| **Map pan & zoom** | Pan/zoom Google Maps until 4 named landmarks are simultaneously visible at zoom ≥ z; smooth-scroll allowed | Continuous feedback (4) + trajectory (1) |
| **Reaction-timed UI** | Dismiss a stack of toast notifications before they auto-fade; click "skip ad" within the window; respond to dropdown autocomplete suggestion within 300 ms | Timing (3) |
| **Game-like minigames** | Catch falling tetris pieces into target slots; piano-roll plays melody at tempo; reaction-time benchmarks | Trajectory + timing (1,3) |
| **Multi-window swap** | While a video plays in window A, transcribe its captions into a doc in window B without missing a sentence | Multi-stream coordination (5) |
| **Voice + GUI** | "Open the highlighted email" while pointing — agent must fuse pointing context with utterance | Multi-stream coordination (5) |

Target: 200 tasks across 9 families, 20–25 per family. Each task has a verifier script that consumes the screencast (or screencast + final state) and outputs pass/fail + a continuous score (e.g. IoU, timing error in ms, completion within deadline).

## Metrics

- **Pass@k** (k = 1, 3, 5) — standard agent metric.
- **Tick latency** — median ms from frame-causing-change to action emitted. Half-duplex baselines will report end-of-loop latency; HBC reports per-tick.
- **Score quality** — continuous score where applicable (drawing IoU, timing error, slider final delta).
- **Mid-action correction rate** — fraction of trajectories where the agent changes plan mid-action in response to UI change. Half-duplex agents are mechanically 0% here.
- **Compute / task** — wall-clock seconds and FLOPs per success.

## Baselines

- Claude Sonnet 4.6 + computer-use API
- Claude Opus 4.5 + computer-use API
- OpenAI CUA / Operator
- Microsoft Fara-7B
- ShowUI (open-source baseline)
- "Oracle half-duplex" — same backbone with infinite screenshot rate but discrete actions, to isolate the action-discretization effect from the observation-rate effect.

## Why this benchmark might be impossible for half-duplex models

For any task in families 1–5, the screenshot-loop is structurally limited:
- A 1s screenshot cycle means the cursor moves blind for ≥ 1s — drag-with-autoscroll over a moving list cannot be solved deterministically.
- Discrete `drag(x1,y1,x2,y2)` cannot trace a non-straight signature path.
- Slider live preview requires observing the histogram while the slider moves; screenshot agent must release, re-screenshot, re-grab — guaranteed overshoot.
- Reaction-timed dismissals demand sub-300 ms loop latency; current agents are at 1–5 s.

The hypothesis: HBC-Bench scores will show <15% for all current half-duplex SOTA, regardless of base model strength. If the paper finds otherwise, the duplex bet is weaker than claimed and we should know.
