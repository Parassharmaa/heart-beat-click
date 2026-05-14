# Heart Beat Click — Ablation Plan

Date: 2026-05-14. Goal: every architectural claim isolated to its own ablation, so reviewers can see which knob earned which gain.

## Core ablations

Each row = same 7B trunk, same data, same total compute. Different architectural switch. Three eval suites: OSWorld-Verified, WebArena, HBC-Bench.

| # | Variant | Duplex tick | Joint W+P loss | Cursor head | Dispatch | Inner monologue | Purpose |
|---|---|---|---|---|---|---|---|
| A0 | Half-duplex screenshot baseline | 1s screenshot loop | policy-only | discrete `click`/`drag` | none | — | grounded SOTA reproduction (Fara-7B-class) |
| A1 | Half-duplex + joint world model | 1s loop | joint | discrete | none | — | does joint loss help w/o duplex? |
| A2 | Duplex 100ms, policy-only | 100ms | policy-only | discrete | none | — | does duplex alone help? |
| A3 | Duplex 100ms + joint W+P (HBC-core) | 100ms | joint | discrete | none | — | duplex + joint together |
| A4 | A3 + flow-matched cursor | 100ms | joint | continuous flow | none | — | adds continuous cursor |
| A5 | A4 + inner monologue | 100ms | joint | continuous flow | none | yes | adds silent reasoning channel |
| A6 | A5 + dispatch to bg reasoner (full HBC) | 100ms | joint | continuous flow | yes (Claude-Opus 4.5 bg) | yes | full system |
| A7 | A6 but bg reasoner = same 7B foreground frozen copy | 100ms | joint | continuous flow | yes (small bg) | yes | does dispatch require frontier bg? |
| A8 | A6 minus dream-RL stage 2 | 100ms | joint | continuous flow | yes | yes | how much does dream RL contribute? |

## Tick-rate sweep (separate axis)

Hold A4 fixed. Vary tick: {1000 ms (≡ screenshot), 500 ms, 200 ms, 100 ms, 50 ms}. Plot HBC-Bench score and OSWorld score against tick rate. Expected: HBC-Bench monotonic improvement to ≤100 ms then plateau; OSWorld near-flat (not tick-sensitive).

## Cursor encoding sweep

Hold A6 fixed. Cursor head variant: {discrete click+drag, FAST-DCT discretized trajectory, OpenVLA-OFT L1 continuous regression, π0-style flow matching}. Eval on trajectory-bound subset of HBC-Bench.

## Background-reasoner sweep

Hold foreground fixed at 7B A6. Background reasoner ∈ {none, Claude Haiku 4.5, Claude Sonnet 4.6, Claude Opus 4.5, GPT-5}. Plot OSWorld score, wall-clock/task, and dispatch-bandwidth (tokens exchanged per task).

## Data ablations

Stages enabled: {S0, S0+S1, S0+S1+S2, all four}. Same A6 backbone. Tests how much each stage contributes.

## Negative controls

- **A0-Plus**: half-duplex screenshot but with same total compute and same data as A6. Tests "is duplex itself responsible, or just the budget?"
- **Random-tick**: same arch as A6 but ticks fire at random 50–500 ms intervals instead of fixed 100 ms. Tests whether the *fixed* clock matters or just the average rate.
- **No-world-prediction at inference**: A6 trained jointly but world head disabled at inference. Tests whether the latent forward model embedded in the trunk is enough.

## Expected outcomes (hypothesis predictions)

| Eval / variant | A0 (screenshot SOTA) | A2 (duplex only) | A3 (duplex+joint) | A6 (full HBC) |
|---|---|---|---|---|
| OSWorld-Verified | 70% (Sonnet 4.6 baseline) | 70% | 73% | 76% |
| WebArena | 65% | 67% | 70% | 73% |
| HBC-Bench | <15% | ~30% | ~40% | **>55%** |
| Trajectory subset of HBC-Bench (sig, slider, scrub) | <5% | ~20% | ~25% | **>50%** |
| Wall-clock per OSWorld task | 60–180 s | 20–60 s | 20–60 s | **15–40 s** |
| Mid-action correction rate | 0% | 5–10% | 10–15% | **20–30%** |

If A6 fails to beat A0 on HBC-Bench by ≥30 pp, the duplex bet is wrong and the paper should not exist.
If A6 ≈ A3 (i.e. dispatch + monologue + flow cursor add little), the simpler A3 is the publishable architecture and the paper should be honest about that.
If A2 ≈ A3 (i.e. joint loss adds nothing), the world-model lens is wrong and we have a smaller story about duplex screencast→action.

## Reviewer-bait ablations

Things a thorough reviewer will demand:

- Replace flow-matched cursor with FAST-DCT discrete cursor — show whichever wins.
- Train HBC backbone from scratch vs warm-start from a text LM vs warm-start from a VLM (Qwen2.5-VL like Fara) — pick the one with cleanest scaling.
- Quantize foreground to fp8 at inference; report any score regression vs latency win.
- Evaluate on AndroidWorld + MobileWorld to show cross-platform transfer.
- Run "infinite-screenshot" half-duplex baseline (10 Hz screenshots, same model size) to *isolate* observation rate from arch — strongest baseline.
