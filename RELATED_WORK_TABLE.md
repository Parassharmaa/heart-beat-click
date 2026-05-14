# Heart Beat Click — Related Work Comparison

Date: 2026-05-14. Paper-style table. Columns = the five HBC commitments. Cell = does this work do that thing? Domain = where it works.

Legend: ✓ = does this; ◐ = does a partial version; ✗ = does not.

| Work | Domain | Full-duplex micro-tick (≤200 ms) | Joint world+policy AR | Continuous flow-matched action head | Foreground/background dispatch | Streaming screencast input |
|---|---|---|---|---|---|---|
| OpenAI CUA / Operator (2025) | GUI / web | ✗ | ✗ | ✗ | ✗ | ✗ (screenshots) |
| Microsoft Fara-7B (2025) | GUI / web | ✗ | ✗ | ✗ | ✗ | ✗ |
| ShowUI (CVPR 2025) | GUI | ✗ | ✗ | ✗ | ✗ | ✗ |
| Claude computer-use (Sonnet 4.6, Opus 4.5; 2025-26) | GUI | ✗ | ✗ | ✗ | ✗ | ✗ |
| ScaleCUA (2025-26) | GUI x-platform | ✗ | ✗ | ✗ | ✗ | ✗ |
| WebWorld (Qwen, 2026) | Web | ✗ | ◐ (world model only, separate from policy) | ✗ | ✗ | ✗ |
| WebSynthesis (2025) | Web | ✗ | ◐ (LLM world model + MCTS, offline) | ✗ | ✗ | ✗ |
| VideoAgentTrek (2025) | GUI pretrain | ✗ | ◐ (predicts steps from video, half-duplex consumer) | ✗ | ✗ | ◐ (video at pretrain only) |
| Moshi (Kyutai, 2024) | Speech | ✓ (12.5 Hz audio, RQ-Transformer) | ✗ | ✗ | ✗ | ✗ |
| NVIDIA PersonaPlex (2026) | Speech | ✓ | ✗ | ✗ | ✗ | ✗ |
| TML-Interaction-Small (2026) | Speech + live video chat | ✓ (200 ms micro-turn) | ✗ | ✗ | ✓ (dual model) | ✗ |
| Gemini 3.1 Flash Live (2026) | Speech + video + tools | ✓ | ✗ | ✗ | ◐ (tool use, not async reasoner) | ✗ |
| π0 / π0.5 (Physical Intelligence, 2024-25) | Robotics | ✗ | ◐ (VLA, world via separate world models) | ✓ (flow matching 50 Hz) | ✗ | ✗ |
| OpenVLA-OFT (2025) | Robotics | ✗ | ✗ | ✓ (L1 continuous + parallel decode) | ✗ | ✗ |
| FAST (Physical Intelligence, 2025) | Robotics | ✗ | ✗ | ◐ (DCT discrete action tokens) | ✗ | ✗ |
| AsyncVLA (2025) | Robotics | ✗ | ✗ | ✓ (async flow matching) | ✗ | ✗ |
| RTC / FASTER (2025-26) | Robotics | ◐ (streaming exec) | ✗ | ✓ | ✗ | ✗ |
| WorldVLA (2025) | Robotics | ✗ | ✓ (AR joint image+action) | ◐ (discrete action) | ✗ | ✗ |
| DreamVLA (2025) | Robotics | ✗ | ✓ (perception-prediction-action loop) | ◐ | ✗ | ✗ |
| Unified Video Action Model (2025) | Robotics | ✗ | ✓ (joint video + action AR) | ◐ | ✗ | ✗ |
| Dreamer 4 (Hafner, 2025) | Games (Minecraft) | ◐ (real-time on 1 GPU) | ✓ (flow-matched world + agent trained inside) | ✓ | ✗ | ✗ |
| Genie 3 (DeepMind, 2026) | 3D worlds | ✓ (24 FPS interactive) | ✗ (world model only) | ✗ | ✗ | ✗ |
| StreamAgent (2025) | Video understanding | ✓ (hierarchical streaming KV) | ✗ | ✗ | ✗ | ✓ (input only) |
| CausVid / Cosmos-Tok / ProVideLLM | Video gen / understanding | ✓ (streaming) | ✗ | ✗ | ✗ | ✓ (input only) |
| Magic Pointer (DeepMind, 2026-05-13) | GUI (augments human) | ◐ | ✗ | ✗ | ✗ | ◐ (cursor region only) |
| **Heart Beat Click (this proposal)** | **GUI** | **✓** | **✓** | **✓** | **✓** | **✓** |

## What the table proves

Every individual ✓ is held by at least one prior work. **No row outside the bottom has all five ✓.** Specifically:

- In the **GUI** column (the domain we care about), no prior work has any of {duplex micro-tick, joint world+policy AR, flow-matched cursor, dispatch}. The closest is Magic Pointer (May 13 2026), which is an *interface* augmentation for a human cursor, not an agent.
- The duplex micro-tick row (Moshi, PersonaPlex, TML, Gemini 3.1 Flash Live) only does speech / audio / generic video chat — not pixel-level GUI control.
- The joint world+policy AR row (WorldVLA, DreamVLA, Unified VA Model, Dreamer 4) only does robotics or games.
- The flow-matched action head row (π0, OpenVLA-OFT, AsyncVLA, RTC, FASTER) only does robot joints.

The HBC contribution is **the row, not the cells**. The paper has to be honest about that.

## Caveat

This table is a snapshot of public literature as of 2026-05-14. Unpublished work at frontier labs (OpenAI, Anthropic, Google DeepMind, xAI, Mistral) may already have unannounced duplex-GUI prototypes. The paper should explicitly acknowledge this. If anything resembling HBC ships from a frontier lab between submission and decision, the contribution becomes the benchmark (HBC-Bench) and the open-source duplex-GUI reference recipe, not architectural priority.
