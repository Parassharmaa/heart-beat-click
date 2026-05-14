# Heart Beat Click — 12-Week Implementation Plan

Date: 2026-05-14. Assumes 2 engineers + 1 researcher, 16-GPU H100 cluster, $30k cloud budget.

## Phase 0 — Foundations (weeks 1–2)

**W1**
- Stand up infra: H100 cluster, browser-VM fleet (200 concurrent), OSWorld + AndroidWorld + WebArena containers, evaluation harness.
- Fork Video2Action / VideoAgentTrek for IDM v0.
- Stand up a "recorder" — instrumented browser + OS that captures synchronized 30-FPS screencast + 60-Hz cursor + key events to disk.
- Hire ~5 contractors for trajectory collection.

**W2**
- Reproduce VideoAgentTrek's reported 15.8% OSWorld-Verified baseline on a Qwen2.5-VL-7B backbone. Pure half-duplex screenshot loop.
- Spin up HBC-Bench v0 with 30 tasks across 3 families (drag-autoscroll, slider, signature). Verifiers + scoring scripts.
- **Deliverable**: reproducible OSWorld baseline + tiny HBC-Bench prototype.

## Phase 1 — Architecture v0 and Stage 0 pretrain (weeks 3–5)

**W3**
- Implement HBC backbone in PyTorch: RQ-Transformer wrapper around Qwen2.5-VL-7B trunk, depth transformer (200M), hMLP screen-patch embed, dMel audio embed (optional v0).
- Implement heads: discrete event AR, FAST-DCT cursor, inner monologue. Flow-matched cursor deferred to W6.
- Streaming KV-session inference harness; verify 200 ms tick at 7B on 1× H100.

**W4–W5**
- Stage 0 pretrain on VideoAgentTrek (10k hours). Joint loss: screen-patch AR (weight 1.0) + IDM-pseudo-label action AR (weight 0.3). Target 26B–60B tokens.
- Cursor-trace recovery: extend Video2Action IDM to emit continuous 60-Hz traces, not just click events. Validate on contractor data held out.
- **Deliverable**: pretrained HBC-7B-S0 checkpoint with measurable world-modeling FVD on held-out screencasts.

## Phase 2 — Stage 1 SFT + first eval (weeks 6–7)

**W6**
- Collect ~500 contractor-hours of high-fidelity duplex trajectories on a target task set drawn from OSWorld + HBC-Bench families.
- Implement flow-matched cursor head (π0 / RTC style). Action chunking 100 ms.
- Stage 1 SFT on contractor data + Fara-7B FaraGen + OS-Genesis trajectories.
- **Deliverable**: HBC-7B-S1 checkpoint.

**W7**
- First full eval pass: HBC-7B-S1 vs Sonnet 4.6, Fara-7B, Operator on OSWorld-Verified + HBC-Bench-v0 + WebArena.
- Tick-rate sweep ablation (50ms, 100ms, 200ms, 500ms, screenshot).
- Cursor-encoding ablation (discrete vs FAST-DCT vs flow-matched).
- **Deliverable**: first results table. Decision gate: do the early signals support H1 (HBC-Bench gap closure) and H4 (cursor encoding)? If both fail, stop and rethink.

## Phase 3 — HBC-Bench v1 and dispatch (weeks 8–9)

**W8**
- Expand HBC-Bench to all 9 families, 200 tasks. Verifiers for every task. Public leaderboard scaffolding.
- Run all baselines on HBC-Bench-v1.
- **Deliverable**: HBC-Bench-v1 release-ready.

**W9**
- Implement dispatch head + non-blocking background-reasoner protocol.
- Background-reasoner options: Claude Opus 4.5 (cloud), Claude Sonnet 4.6 (cloud, cheap), or self-hosted Qwen3-72B.
- Run dispatch ablations (none / haiku / sonnet / opus / GPT-5).
- **Deliverable**: HBC-7B-S1+Dispatch checkpoint and dispatch ablation results.

## Phase 4 — Stage 2 dream-RL (weeks 10–11)

**W10**
- Implement dream-rollout pipeline: roll out N=16 candidate trajectories in latent world model, score via inner-monologue confidence + value head, pick best, replay in real browser, compute advantage.
- Verify dreams are not catastrophically hallucinatory: visual FID between dream and real next-frame on held-out trajectories.

**W11**
- Run dream-RL on 5k browser tasks. Update policy + (optionally) world model from advantage.
- Stage 3 RLHF can be deferred to post-paper; the science contribution does not depend on it.
- **Deliverable**: HBC-7B-S2 checkpoint, dream-RL ablation A8 vs A6.

## Phase 5 — Paper writing and red-teaming (week 12)

**W12**
- Final eval sweep across all checkpoints, all ablations, all eval suites.
- Generate all 7 figures.
- Write paper following PAPER_OUTLINE.md.
- Red-team week: deliberately try prompt-injection attacks via injected page content, malicious system audio, fake UI elements. Document mitigations.
- Internal safety review.
- **Deliverable**: arxiv preprint + open-source release of HBC-Bench eval harness + Stage 0 + Stage 1 checkpoints.

## Critical-path risks and contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| 100 ms tick unachievable at 7B | medium | fall back to 200 ms tick; still beats 1s screenshot loop |
| Cursor IDM recovery noisy → bad action labels | medium | rely more on contractor data; reduce S0 action-loss weight |
| Dream rollouts collapse / reward-hack | medium | mandatory real-browser executor verification; throw out unverified updates |
| Background reasoner adds wall-clock latency instead of removing | low | strict timeouts, foreground never waits, log abandonment rate |
| Contractor data insufficient | medium | augment with WebSynthesis tree-searched synthetic data; supplement with AgentTrek |
| HBC-Bench accidentally solvable by half-duplex with extra screenshots | high (need to verify) | A0-Plus baseline (infinite screenshot rate); discard tasks where half-duplex solves >40% |
| Prompt-injection attacks exploit duplex speed | high | revocable-action gating; confirmation on irreversible; see SAFETY.md |

## Decision gates

- **End of W7**: if H1 (HBC-Bench gap) or H4 (cursor encoding) signal is missing on the v0 benchmark, hard stop and reformulate. No point burning W8–W12 budget chasing dead hypothesis.
- **End of W9**: if dispatch adds latency rather than removing it, drop it from v1 paper and ship simpler arch.
- **End of W11**: if dream-RL gives <5 pp improvement on any eval, drop it from v1 paper; future-work it.

## After v0

- v1: scale to 14B or larger MoE.
- v1: add Stage 3 RLHF / barge-in.
- v1: cross-platform (Android, macOS, Windows) eval.
- v2: agent-to-agent duplex coordination — two HBC agents collaborating on a shared screen.
