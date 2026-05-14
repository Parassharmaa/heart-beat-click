# Heart Beat Click

Full-duplex computer-use agent. Screencast streams in, cursor + event streams out, both continuous, on a 100 ms tick.

Inspired by Thinking Machines' TML-Interaction-Small (May 2026) and Kyutai's Moshi. Combined with Dreamer 4's flow-matched world model and π0's continuous action heads, retargeted at pixel-level GUI control.

## Repo guide (read in this order)

| File | What it answers |
|---|---|
| [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) | One-page brief. 30-second read. Start here. |
| [ABSTRACT.md](ABSTRACT.md) | Paper abstract v1. |
| [LITERATURE.md](LITERATURE.md) | What exists in duplex models, computer-use agents, VLA heads, video tokenizers, world models. Cites everything. |
| [NOVELTY.md](NOVELTY.md) | Adversarial check of every claim against existing work. Honest about what's borrowed and what's new. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full architecture proposal: streams, backbone, heads, dispatch, training stages, risks. |
| [HBC_BENCH.md](HBC_BENCH.md) | New benchmark of 200 tasks across 9 families where half-duplex agents are structurally limited. |
| [COMPUTE.md](COMPUTE.md) | Per-tick token budget, latency reality-check vs Moshi/TML, training cost ~7k H100-hours. |
| [DATA.md](DATA.md) | Four-stage pretraining recipe with concrete corpora (VideoAgentTrek, MONDAY, ScaleCUA, contractor logs). |
| [ABLATIONS.md](ABLATIONS.md) | A0–A8 ablation table + sweeps + falsification criteria. |
| [PAPER_OUTLINE.md](PAPER_OUTLINE.md) | Section-by-section paper plan + 7 figures + reviewer objections answered. |
| [MILESTONES.md](MILESTONES.md) | 12-week implementation plan with weekly deliverables and decision gates. |
| [SAFETY.md](SAFETY.md) | Threat model both expanded and contracted by duplex. Required mitigations. Pre-release red-team protocol. |
| [CODE_SKETCH.md](CODE_SKETCH.md) | PyTorch-style pseudocode for streams, RQ backbone, heads, tick loop, joint loss, dream-RL. |
| [RELATED_WORK_TABLE.md](RELATED_WORK_TABLE.md) | Paper-style comparison of 25+ works against HBC's 5 commitments. Visual proof the intersection is empty. |

## One-line thesis

The first joint AR world-and-policy model with a Moshi-class duplex micro-tick loop applied to pixel-level GUI control.

## Five hypotheses (testable)

1. **H1**: half-duplex SOTA scores <15% on HBC-Bench at any model scale; a 7B HBC exceeds 50%.
2. **H2**: unified joint world+policy matches separate-network world-model lookahead at parity FLOPs.
3. **H3**: dispatch yields >10× lower wall-clock per OSWorld task at parity success.
4. **H4**: flow-matched continuous cursor beats discrete `click(x,y)` on trajectory-bound tasks by ≥30 pp.
5. **H5**: Dreamer-4-style in-model dream RL reaches Fara-7B WebVoyager with <10% of its trajectory data.

## Honest scope

Every individual ingredient — duplex micro-tick (Moshi/TML), joint world+action (Dreamer 4 / WorldVLA), flow-matched action head (π0), dispatch (TML), screencast pretraining (VideoAgentTrek) — exists in published literature. The contribution is the intersection applied to GUI, plus a benchmark that measures what current half-duplex models cannot do regardless of scale.

## Status

This is a research proposal, not an implementation. v0 implementation budget: 12 weeks, ~$30k, 16-GPU H100 cluster (see MILESTONES.md).
