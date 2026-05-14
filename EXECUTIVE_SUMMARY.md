# Heart Beat Click — Executive Summary

Date: 2026-05-14. One-page brief for a 30-second read.

## What

Full-duplex computer-use agent. Screencast streams in at 10–25 FPS, cursor (continuous, flow-matched) + key/scroll events (discrete) stream out at 100 ms tick. Same network predicts next screen frame *and* next action. Foreground tick model never blocks; offloads slow reasoning to async background reasoner.

## Why

Speech crossed the duplex chasm in 2024–2026 (Moshi → PersonaPlex → TML-Interaction-Small). GUI did not. Half-duplex screenshot agents (CUA, Fara-7B, Sonnet 4.6) are near human on OSWorld (72.5% vs 70–75%) but structurally blind to animation, drag, scrub, slider, timing — the remaining error mass.

## How (architecture)

- Encoder-free early fusion (TML pattern): hMLP screen patches, dMel audio, learned event embeddings — no separate pretrained vision tower.
- RQ-Transformer (Moshi pattern): 7B temporal trunk + 200M depth trunk over 5 codebooks {cursor, event, monologue, dispatch, screen-pred}.
- Flow-matched cursor head (π0 / Dreamer-4 pattern) at 60 Hz inside each 100 ms tick.
- Joint world-model + policy loss (WorldVLA / Dreamer 4 pattern) → in-model dream rollouts for RL.
- TML dual-model dispatch: foreground 7B at 100 ms tick, background Opus-class reasoner async via dispatch channel.

## Five hypotheses (falsifiable)

| # | Claim | Falsified if... |
|---|---|---|
| H1 | Half-duplex SOTA < 15% on HBC-Bench; HBC-7B > 50% | gap < 30 pp |
| H2 | Unified world+policy = separate-network lookahead at parity FLOPs | unified strictly worse |
| H3 | Dispatch yields > 10× lower wall-clock at parity OSWorld success | < 3× |
| H4 | Flow-matched cursor > discrete clicks by ≥ 30 pp on trajectory-bound tasks | < 10 pp |
| H5 | Dream-RL reaches Fara-7B on WebVoyager with < 10% data | needs > 30% |

## Honest novelty

Every ingredient exists. Duplex micro-tick (Moshi/TML, speech), joint world+action AR (Dreamer 4 / WorldVLA, games/robots), flow-matched action heads (π0, robots), dispatch (TML, speech), screencast pretrain (VideoAgentTrek). **Intersection in GUI is empty.** HBC is the union; HBC-Bench is the benchmark that measures what current architectures *can't* do at any scale.

## Cost

~7k H100-hours total over 4 stages (S0 screencast pretrain → S1 SFT → S2 dream-RL → S3 RLHF). ≈ $25–30k cloud. 12 weeks with 2 engineers + 1 researcher + 16-GPU H100 cluster. Reproducible outside frontier labs.

## Decision gates (kill switches)

- W7: if H1 or H4 unsigned on HBC-Bench-v0, stop.
- W9: if dispatch adds latency rather than removing, drop from v1.
- W11: if dream-RL gain < 5 pp, drop and future-work it.

## Files (this repo)

- Paper: `ABSTRACT.md`, `PAPER_OUTLINE.md`
- Architecture: `ARCHITECTURE.md`, `CODE_SKETCH.md`
- Survey: `LITERATURE.md`, `NOVELTY.md`, `RELATED_WORK_TABLE.md`
- Benchmark: `HBC_BENCH.md`
- Plan: `COMPUTE.md`, `DATA.md`, `ABLATIONS.md`, `MILESTONES.md`
- Safety: `SAFETY.md`
- This summary: `EXECUTIVE_SUMMARY.md`

## Next decisions for the user

1. **Is this the right problem?** If yes, the package is ready to share with a co-author / advisor for critique.
2. **Pursue v0 implementation?** If yes, MILESTONES.md → start at W1 actions (infra + recorder).
3. **Reframe as a benchmark-first paper?** If unwilling to commit 12 weeks, ship HBC-Bench as a standalone evaluation contribution; let other labs train on it.
4. **Reorient any of the 5 hypotheses?** Currently all five are testable; some may be more valuable than others to a specific venue.

Loop self-paces until redirected. Tell me what to do next.
