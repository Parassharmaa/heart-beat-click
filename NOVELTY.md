# Heart Beat Click — Adversarial Novelty Audit

Date: 2026-05-14. Goal: stress-test the "new architecture" claim against what already exists.

## What was claimed in v0 abstract

1. Full-duplex micro-tick screencast↔action loop
2. Joint world-model + policy objective (predict next frame and next action)
3. Flow-matched continuous cursor head
4. RQ-Transformer multi-codebook action stream
5. Dual-model dispatch (foreground tick + background reasoner)

## Adversarial findings (chronological)

| Year | Work | What it does | Threat to which claim |
|---|---|---|---|
| 2024-10 | **Moshi** (Kyutai) | RQ-Transformer + dual streams + inner monologue for **speech** | Validates (4); domain ≠ GUI, no threat |
| 2024-10 | **π0** (Physical Intelligence) | Flow-matching action head for **robotics** | (3) no longer novel as a head; novel for cursor specifically |
| 2025-02 | **UP-VLA** | Joint multi-modal + future-state objective for robotics | (2) no longer novel for robotics |
| 2025-06 | **WorldVLA** | AR joint action + image gen for robotics, "single framework" | (2) **directly precedent for joint world+action** in one AR backbone — but robotics camera/joints, not screencast/cursor |
| 2025-07 | **DreamVLA** | World-knowledge forecasting + inverse dynamics + perception-prediction-action loop, robotics | (2) more precedent for joint world+action in VLA |
| 2025-07 | **WebSynthesis** | LLM world model + MCTS to synth WebUI trajectories offline | (2) **closest GUI precedent for using a world model with a web agent** — but offline trajectory *synthesis*, not duplex policy; world model and policy are separate; half-duplex screenshot consumer |
| 2025-09 | **Dreamer 4** (Hafner) | Flow-matching world model with "shortcut forcing", real-time on 1 GPU, trains agent inside world model, Minecraft diamond from offline-only with 100× less data than VPT | (2) **strongest precedent for joint flow-matched world model + agent training**, but games not GUI, and policy is a separate head trained inside the model rather than co-AR with frames |
| 2026-01 | **Genie 3** (DeepMind) | 24 FPS real-time interactive 3D world model | (2) world-model side; not joint with action policy in one net |
| 2026-01 | **PersonaPlex-7B** (NVIDIA) | Moshi-arch duplex speech | Validates (1) for speech; not GUI |
| 2026-01 | **WebWorld** (Qwen) | 8B/14B/32B web world model, +10.9% WebArena as separate net | (2) GUI world model; explicitly separate from policy |
| 2026-05-11 | **TML-Interaction-Small** | Encoder-free early fusion + 200 ms micro-turns + foreground/background dual model for **speech+video chat** | Validates (1), (5); not GUI agent |
| 2026-05-13 | **DeepMind Magic Pointer** | Gemini-powered AI cursor that captures visual + semantic context around user's cursor | Adjacent: AI augments *human* pointer, not an autonomous agent — does not threaten any HBC claim, but signals the cursor is becoming an AI-aware surface |
| 2026-02 | **CoAct-1** | Multi-agent GUI Operator + Programmer with high-level Orchestrator dispatching subtasks | Higher-level dispatch among separate agents (process boundary). HBC dispatch is intra-loop (token-level, non-blocking, shared trunk). Different mechanism, similar motivation. |
| 2026-05-07 | **Grok Computer** (xAI) | Half-duplex desktop control agent | Half-duplex, no threat to (1) |
| 2026-05 | **Perplexity Computer for Enterprise** | Multi-model orchestrator + half-duplex desktop control | Half-duplex, no threat to (1) |

### Half-duplex GUI agents (the ones HBC competes with)
- OpenAI CUA, Microsoft Fara-7B, ShowUI, Claude Sonnet 4.6 (72.5% OSWorld), Claude Opus 4.5 (66.3% OSWorld). All screenshot-loop. None duplex. None joint world+policy in one net.

## What remains genuinely novel

Each isolated ingredient is taken. **The intersection is empty.** Specifically, no published work satisfies all of:

1. **Full-duplex sub-second screencast→action loop** (Moshi/TML do speech; Fara/CUA do screenshots, not streams).
2. **Joint world-model + policy in one AR backbone on GUI** (WorldVLA/Dreamer 4 do this for robotics/games; WebWorld and WebSynthesis split world model from policy on GUI).
3. **Flow-matched continuous cursor head** (π0 has the head for robot joints; nobody points it at GUI cursors).
4. **TML-style dual-model dispatch ported to GUI** (TML demonstrates for speech; nobody for GUI).
5. **Operating at a tick rate that makes drag / animation / debounce / scroll-inertia first-class observations** (OSWorld's drag is a discrete primitive, not a streamed trajectory).

The defensible thesis is **the unified GUI-side instantiation**, not any single ingredient. The paper has to be honest about that, frame v1 as "Dreamer 4 ∪ TML-Interaction ∪ π0 ∪ Moshi, retargeted at screencast and cursor", and earn its novelty through:

- the GUI domain (none of the precedents),
- the duplex micro-tick loop (none of the GUI precedents have it),
- the joint AR co-prediction of frames and continuous cursor trajectories (Dreamer 4 has joint, but separate flow-matching policy heads; WorldVLA has joint AR, but for robot images and discrete joints),
- the new benchmark that exposes what current half-duplex models can't do.

## Headroom argument

OSWorld is closing: Claude Sonnet 4.6 hits 72.5% vs human 70–75%. Adding more screenshot-loop reasoning is hitting diminishing returns. The remaining error mass on OSWorld is concentrated in tasks where the **observation stream itself is the bottleneck**: animations, drags, video scrubbing, autocomplete debounce, slider control, modal latency, multi-window timing. Half-duplex agents are structurally blind to these. This is the wedge where a duplex architecture is not just nicer — it's necessary.

Hence the need for **HBC-Bench** (see HBC_BENCH.md), separate from OSWorld, to evaluate the part of the task space where the duplex bet actually pays.

## Updated novelty claim (for revised abstract)

> "We do not claim that joint world-model + policy training is itself novel — recent robotics (WorldVLA, DreamVLA, π0.5) and game-agent (Dreamer 4) work have established this paradigm. Nor is duplex micro-tick architecture novel — speech-side systems (Moshi, PersonaPlex, TML-Interaction-Small) have demonstrated it. We claim novelty in the *combination*: the first joint AR world+policy model with a Moshi-class duplex micro-tick loop applied to pixel-level GUI control, with a continuous flow-matched cursor head and a TML-style foreground/background dispatch. We further contribute HBC-Bench, a benchmark of time-sensitive GUI tasks where half-duplex screenshot-loop agents are structurally limited regardless of model scale."
