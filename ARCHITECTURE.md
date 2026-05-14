# Heart Beat Click — Architecture Proposal

A full-duplex computer-use agent. Screencast streams in, actions stream out, both continuously. Inspired by Thinking Machines' Interaction Models, Moshi's RQ-Transformer, π0's flow-matching action head, and WebWorld's world-model objective.

## 0. Naming

- **Tick** = one 100 ms interaction step (the "heart beat"). Configurable 50–200 ms.
- **Click** = any motor token emitted in a tick (mouse, key, scroll, drag-step).
- **Screencast** = causal video stream of the user's display, sampled at 10–25 FPS.

## 1. Inputs and outputs per tick

Every tick the model consumes one chunk of input streams and emits one chunk of output streams. All streams are time-aligned and interleaved (Moshi / TML pattern).

Input streams:
- **Screen patches**: 40×40 pixel patches (TML hMLP style) of the latest frame chunk for the tick. Causal video tokenizer (Cosmos-style temporal causal conv) holds rolling state, so re-encoding is incremental.
- **Cursor proprioception**: real (x, y, button-state) at OS-event rate, bucketized to the tick. Tells the model where its own cursor *actually* is, not where it last asked it to go.
- **DOM/a11y patch deltas** (optional auxiliary): only deltas since last tick, encoded as short text tokens. Helps grounding without paying full-tree cost.
- **Audio** (optional): system audio + mic, dMel-embedded, for video/voice apps.
- **User text / voice**: instruction channel.

Output streams (parallel codebooks, RQ-Transformer style):
- **Cursor head** (continuous): flow-matched Δ(x, y) trajectory chunk per tick — π0 / OpenVLA-OFT style. Continuous, smooth, supports drag/draw/scroll-inertia.
- **Event head** (discrete): key-down, key-up, click, scroll-wheel notch, hotkey, app-switch. RQ codebook per tick.
- **Inner-monologue text head** (discrete, silent): Moshi-style text track running parallel to motor streams. Reasoning bandwidth without blocking motor output.
- **Dispatch head** (discrete): special tokens that hand a sub-task to the Background Reasoner (see §4).

## 2. Backbone

Two transformers in the spirit of Moshi's RQ-Transformer:

- **Temporal Transformer (large)**: causal, sees the time axis of all interleaved tokens. Hosts most params. KV-cached as a streaming session (TML's persistent-GPU-sequence trick). MoE-ready; the foreground model targets ~7–14B active for tick-budget viability — TML themselves note "larger pretrained models are currently too slow to serve in this setting."
- **Depth Transformer (small)**: per-tick, models inter-channel dependence across {cursor, event, monologue, dispatch} codebooks. Lets us emit multiple modality tokens per tick without bloating sequence length.

Early fusion, encoder-free (TML's design choice): no separately pretrained vision tower. hMLP for image patches, light dMel for audio, learned token embeddings for events. Everything trained jointly from scratch (or from a text-LM init for the temporal trunk).

## 3. Joint world-model + policy objective

Train next-tick prediction over **both** screen-patch tokens **and** action tokens (cursor + event). This is the central novelty.

Loss = α · L_action + β · L_screen + γ · L_monologue (+ aux L_dom_delta).

Consequences:
1. **Visual proprioception**: the model is forced to predict what the screen will look like *given* its planned actions. Latent forward model emerges as a free byproduct.
2. **Dreaming**: at inference time, the temporal trunk can be rolled forward without grounding to generate hypothetical futures — drop-in lookahead search (cf. WebWorld used as inference-time world model beating GPT-5 on web tasks).
3. **Self-play training**: generate trajectories inside the model's own dream rollouts, then verify via a real-browser executor (FaraGen-style). Closes the data flywheel.
4. **Continuous timing**: because frame prediction is at tick rate, the model learns animation duration, autocomplete debounce, modal-open latency — things current screenshot agents are blind to.

This is the direct GUI analogue of the Unified Video Action Model for robotics, but with screencast and OS events instead of camera and joints, and with the duplex micro-tick loop layered on top.

## 4. Dual-model dispatch (TML's Foreground/Background pattern, ported)

- **Foreground (HBC tick model)**: must finish a forward pass within the tick budget. Owns motor control, monologue, perception. ~7–14B active.
- **Background Reasoner**: large reasoning model (frontier-class, can be off-device). Receives a packed dispatch context written by the foreground via the dispatch head, runs slow CoT / web search / planning, and streams plan tokens back into a dedicated **plan-injection channel** the foreground reads on subsequent ticks.
- Foreground never blocks waiting; it carries on with the last good plan until new plan tokens arrive (TML notes "results stream back to the frontend, which inserts them into the dialogue at a natural moment").

This solves the latency–IQ tradeoff that currently forces computer-use agents to choose between Operator-grade reasoning (slow) and Fara-grade tick rate (shallow).

## 5. Action representation details

The cursor head is the architecturally interesting part because GUI cursor traces are higher-frequency than typical robot arms (often 60–120 Hz raw mouse events) but lower-dimensional (2D + buttons).

Two viable encodings:

- **Flow-matched continuous chunk**: predict the next 100 ms of cursor delta as a small continuous trajectory, π0 / AsyncVLA style. Native to drag, draw, smooth scroll.
- **FAST-DCT discrete chunk**: DCT-compress the trajectory chunk, tokenize coefficients. Drops into an AR LM cleanly, no separate head needed. Probably the better v0.

Event head stays discrete-AR; events are sparse, semantic, low-rate.

Cursor head emits at 50 Hz (sub-tick); event head emits at tick rate; monologue head emits opportunistically.

## 6. Streaming-session inference

Borrow TML's infra wholesale:
- Persistent GPU sequence per session, no realloc per tick.
- MoE gather+gemv kernels for bidirectional serving.
- Batch-invariant kernels so the on-device runtime matches the trainer bit-for-bit (matters for RL fine-tune stability).
- Per-tick budget = (encode patches + temporal trunk forward + depth trunk forward + decode action chunk). Target 60 ms on a single H100-class GPU for 7B foreground.

Failure-mode notes from TML's blog apply here too:
- Long sessions bloat context — need sliding window + episodic compression (StreamAgent's hierarchical KV-cache is the obvious reference).
- Network unreliability degrades the Background Reasoner — design degradation must be graceful (foreground keeps running on stale plan).

## 7. Data and training stages

**Stage 0 — pretrain on screencast world-model objective only**: massive corpus of public screencasts (YouTube tutorials, OS demos, app reviews) with synthetic cursor + event labels recovered by inverse-dynamics labeller. Predict next frame chunk only. Free data, no agent needed.

**Stage 1 — supervised duplex on real trajectories**: real user sessions and Operator/Fara-style logged trajectories. Now action heads are trained alongside the screen head. FaraGen-style multi-agent verified data extends coverage.

**Stage 2 — dream RL**: model rolls itself out in its own world-model dream, executes the dream's "best" trajectory in a real browser sandbox, scores against task, gradient-updates on advantage. WebWorld already showed +10.9% WebArena gain just by being a separate world model — having world model and policy be the *same* network is strictly more efficient.

**Stage 3 — duplex RLHF / barge-in**: human users interrupt mid-action ("no, click the other one") to teach turn-taking and revocation. Same trick that made Moshi's barge-in work.

## 8. Why this is plausibly new

| Ingredient | Exists | Used here for GUI? |
|---|---|---|
| Encoder-free early fusion (TML) | yes — speech/video chat | not for GUI |
| Micro-tick duplex loop (TML, Moshi) | yes — speech | not for GUI |
| RQ-Transformer (Moshi) | yes — speech | not for screencast+action |
| Flow-matched action head (π0, OpenVLA-OFT) | yes — robotics | not for cursor / GUI |
| FAST DCT action tokens | yes — robotics | not for cursor |
| Causal streaming video tokenizer (Cosmos, CausVid, ProVideLLM) | yes — gen video | not in a duplex agent |
| Joint video+action AR (Unified VA Model) | yes — robotics | not for GUI |
| Foreground/Background dispatch (TML) | yes — speech | not for GUI |
| World-model lookahead for web (WebWorld) | yes — but as separate net | not jointly with policy |

Heart Beat Click = stitch them all into one duplex GUI agent. Each ingredient is published; the integration and the GUI domain are open.

## 9. Hypotheses (testable)

- **H1 — Duplex closes the time-sensitive task gap.** Tasks involving animations, drag, scroll inertia, autocomplete, video timing, mid-modal dialogs are <20% solved by current screenshot-loop agents on OSWorld. HBC's tick loop should push this above 50% with no extra reasoning, just from observation bandwidth.
- **H2 — Joint world+policy beats separate.** Same compute budget, ablated. The unified network should match WebWorld-as-lookahead while being 1 model not 2.
- **H3 — Dispatch unlocks the IQ/latency Pareto.** Foreground tick at 100 ms + Background reasoner async should match Operator's task success on OSWorld at <1/10 the wall-clock per task.
- **H4 — Flow-matched cursor > discrete click tokens** for any task needing trajectory (drawing, drag-to-resize, slider, map-pan), measured by task success on a new HBC-Motor benchmark.
- **H5 — Dream RL is data-efficient.** A 7B HBC fine-tuned via in-model dream rollouts + real-browser verification matches Fara-7B on WebVoyager with <10% of the trajectory data.

## 10. Risks and unknowns

- 100 ms tick budget on 7B+joint-world-model is tight; may need to drop video head at inference and run it only at train time (still buys the proprioception).
- Encoder-free early fusion from scratch is expensive; warm-starting the temporal trunk from a text LM may break visual learning.
- Dream rollouts in pixel space will hallucinate and the policy will exploit hallucinations. Real-browser verification gate is mandatory; pure-dream RL will collapse.
- High-freq cursor data is heavy on disk — DCT compression (FAST) at recording time is probably required.
- Background Reasoner channel risks becoming the bottleneck if foreground waits on it; protocol must be strictly non-blocking with timeouts and a "carry on" default.
