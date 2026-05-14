# Heart Beat Click — Data Sourcing & Pretraining Plan

Date: 2026-05-14. Maps existing corpora and IDM techniques to HBC's four training stages.

## Stage 0 — Self-supervised screencast world-model pretrain

**Goal**: learn to predict next frame chunk conditioned on past frames + past actions; learn the OS/web/app dynamics from raw video.

**Corpora (already published, public)**:
- **VideoAgentTrek** (Oct 2025): 55k screen-capture videos, 10k hours raw, 7,377 hours filtered by ScreenFilter (YOLO-based cursor detector). 39k videos → 1.52M ReAct steps → 26B tokens. Tutorials dominate 69.6%. This is the obvious base. Their Video2Action IDM detects click(x,y), drag-path, scroll-Δ, typed-text with temporal bounds.
- **MONDAY**: 313K annotated frames from 20K YouTube mobile-OS instructional videos. Mobile coverage.
- **ScaleCUA**: cross-platform corpus across 6 OS and 3 task domains. Heterogeneity.
- **Raw YouTube screencast tag**: large untapped reservoir if we run our own IDM. VPT showed 2k labeled hours of contractor data → IDM → 70k pseudo-labeled hours. Apply same recipe to screencast: ~500–2k hours of consensual user-instrumented sessions → IDM → tens of thousands of hours of public screencast.

**Action labelling pipeline (HBC-IDM)**:
- Re-use Video2Action as drop-in for v0.
- Augment with **continuous cursor trajectory recovery**: existing IDMs emit `click(x,y)` events, but HBC needs full 60 Hz cursor traces. Train a cursor-localization head (Magic Pointer's "pixels-to-entities" idea applies in reverse — find the AI's pointer in video) + temporal smoothing → continuous trace.
- For unrecorded videos (no cursor visible), use synthetic-cursor inference: predict cursor from window-focus + click sparks + caret blinks.
- For DOM-rich videos (browser tutorials), align reconstructed DOM tree to frames; gives free grounding labels.

**Loss**: AR over interleaved {screen-patch tokens, cursor head, event tokens, monologue tokens}. Frame-prediction weight large; action-prediction weight small (labels are noisy IDM pseudo-labels).

## Stage 1 — Duplex SFT on logged trajectories

**Goal**: supervise the action heads with high-quality trajectories where both screen *and* exact action are known.

**Corpora**:
- **Operator / CUA traces**: closed; obtainable only via OpenAI partnership. Probably not available.
- **Fara-7B's FaraGen output**: 145k browser trajectories, 1M steps, public via Microsoft. Verified by their multi-agent pipeline. Clean labels but screenshot-rate (not 100 ms tick).
- **AgentTrek / AgentTrek-7B**: 20k tutorial-guided synthetic trajectories.
- **OS-Genesis-7B trajectories**: 7.4k real-world trajectories.
- **WebSynthesis**: 4k synthetic but tree-searched-good trajectories.
- **Self-collected via instrumented browser/OS recorder**: contractors do tasks while we log frames @ 30 FPS + raw `xinput`/Win32/CGEvent stream. ~500–1000 contractor-hours for v0 → ~50k high-fidelity tick-rate trajectories.

**Label format**: every trajectory has synchronized {screencast @ 30 FPS, cursor xy @ 60 Hz, keyboard events, scroll deltas, app focus, task description, success/fail}. Re-sample to HBC tick (100 ms).

**Loss**: Stage 0 loss + a stronger weight on action heads + a behavioral-cloning regularizer.

## Stage 2 — In-model dream RL with executor verification

**Goal**: improve policy beyond the imitation ceiling using the model's own world-model rollouts.

**Recipe (after Dreamer 4, Hafner 2025)**:
1. Sample a task description + start screen.
2. Roll out N candidate action sequences inside the model's latent world model (frame head + action head co-fire; no real browser).
3. Pick the dream where the model's monologue head predicts highest task-completion confidence.
4. Replay that action sequence on a real browser / VM.
5. Executor produces ground-truth success and grounded next-frames.
6. Compute advantage = real-success − dream-predicted-success. Gradient on policy (and on world model where executor disagrees with dream).

Dreamer 4 hit Minecraft diamond from offline-only with 100× less data than VPT. The analogous bet for GUI: dream RL on a 7B HBC reaches Fara-7B WebVoyager performance with <10% of FaraGen data.

**Executor stack**:
- WebArena container fleet (~200 concurrent VMs) for web tasks.
- OSWorld VMs for desktop tasks.
- AndroidWorld emulators for mobile tasks.
- HBC-Bench env for time-sensitive tasks.

**Verification**: the executor itself; no LLM-as-judge in the loop (LLM-as-judge bleeds reward hacking risk).

## Stage 3 — Duplex RLHF with mid-action barge-in

**Goal**: teach the model to (a) accept correction mid-action, (b) yield turn gracefully, (c) prefer revocable actions when uncertain, (d) ask for confirmation on irreversible ops.

**Method**: Moshi-style. Real human user does live tasks. At any moment they can talk over the agent ("no, the other button") or type a correction. We collect preference pairs over {trajectory-as-completed, trajectory-with-correction} and DPO/GRPO-train.

Volume: 10k–50k preference comparisons. Cheap-ish.

## Bonus — synthetic dataset for HBC-Bench training

HBC-Bench tasks (signature, drag-autoscroll, slider, scrub, ...) need their own training data. Approach:
- Generate procedural tasks at scale: 100k random sign-drawing targets, slider-target combinations, draggable-list configurations.
- Each task ships with a script-controlled "expert demonstration" — programmatic cursor moves that solve the task within the time budget.
- HBC supervises on these. Then dream-RL refines.

## Data risks / mitigations

| Risk | Mitigation |
|---|---|
| IDM pseudo-labels are noisy → policy learns to imitate IDM artifacts | Stage 1 SFT on high-fidelity contractor data corrects |
| YouTube screencast license / consent issues | Use VideoAgentTrek's already-published corpus; for own crawls, consent-by-channel-owner |
| Self-collected screencasts capture personal data | Run capture inside disposable VMs only; redact OCR-detected PII |
| Distribution shift from training apps to deployment apps | Stage 0 cross-platform corpus + Stage 2 dream-RL on target domains |
| Action label drift (e.g. cursor disappears in some apps) | Maintain dual labelling: pixel-IDM and OS-event-log; reconcile |
