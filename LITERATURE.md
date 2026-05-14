# Heart Beat Click — Literature Review

Date: 2026-05-13. Project: full-duplex computer-use agent (screencast in, action out, realtime). Inspired by Thinking Machines' Interaction Models.

## 1. Full-duplex multimodal models (the seed idea)

### Thinking Machines — TML-Interaction-Small (May 12 2026)
- 276B MoE, 12B active.
- **Encoder-free early fusion**: no pretrained encoders. Audio → dMel + light embedding. Video → 40×40 patches → hMLP. Audio out → flow head. All trained jointly from scratch with the transformer.
- **Micro-turn loop**: 200ms in, 200ms out, interleaved streams. Continuous duplex, not turn-based.
- **Dual-model**: Foreground "Interaction" model keeps dialogue live. Background model handles slow reasoning + tools, streams results back in.
- Latency 0.40s end-to-end on FD-bench. Gemini-3.1-flash-live 0.57s, GPT-realtime-2.0 1.18s.
- Streaming sessions: persistent GPU KV. MoE kernels gather+gemv for bidirectional serving. Batch-invariant kernels for trainer-sampler bit alignment.
- Known failure modes: long-context bloat, network reliance, can't scale larger model under tick budget.
- Source: https://thinkingmachines.ai/blog/interaction-models/

### Moshi (Kyutai, Oct 2024) — the precedent
- RQ-Transformer: large **Temporal Transformer** (7B) over time + small **Depth Transformer** over codebook channels per timestep.
- Mimi neural codec, streaming.
- **Parallel streams** for user audio + model audio (so it can listen while talking).
- **Inner monologue**: silent text token stream parallel to audio stream → reasoning channel that doesn't cost audio bandwidth.
- Source: https://arxiv.org/abs/2410.00037

### NVIDIA PersonaPlex-7B (Jan 2026)
- Moshi-style: Mimi encoder → Temporal+Depth transformer → Mimi decoder. 7B params.
- Continuous codec audio, joint text+audio token prediction. Barge-in, overlap, fast turn-taking native.
- Source: https://research.nvidia.com/labs/adlr/personaplex/

### SALM-Duplex (May 2025)
- Continuous user input + codec agent output. Channel fusion. 0.6 kbps agent voice. Separate user/agent arch.
- Source: https://arxiv.org/abs/2505.15670

## 2. Computer-use agents (current SOTA — all half-duplex)

### OpenAI CUA (Operator)
- GPT-4o vision + RL. Screenshot → reason → tool-call action → wait → next screenshot.
- OSWorld 38.1%, WebArena 58.1%, WebVoyager 87%.
- Source: https://openai.com/index/computer-using-agent/

### Microsoft Fara-7B (Nov 2025)
- Decoder-only multimodal LM, Qwen2.5-VL-7B base. 128k ctx.
- Pixel-only (no a11y tree). Screenshot → CoT → tool call with coords.
- ~124k input / 1.1k output tokens per WebVoyager task, 16.5 actions, $0.025/task.
- Trained on 145k browser trajectories / 1M steps from FaraGen pipeline.
- Source: https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/

### ShowUI (CVPR 2025)
- One VLA for GUI. **Interleaved vision-language-action streaming**. UI-guided visual token selection: −33% tokens, 1.4× train speed.
- Source: https://arxiv.org/abs/2411.17465

### Survey of ACUs
- Source: https://arxiv.org/abs/2501.16150

## 3. Vision-Language-Action heads — continuous action prediction

### π0 / π0.5 (Physical Intelligence)
- VLM backbone + **flow-matching action head** → continuous smooth trajectories, up to 50 Hz dexterous control.
- Sources: https://arxiv.org/abs/2410.24164, https://www.pi.website/download/pi05.pdf

### OpenVLA-OFT
- Parallel decoding + action chunking + continuous action + L1 regression. 25-50× faster gen, 3× lower latency.
- Source: https://openvla-oft.github.io/

### AsyncVLA
- Asynchronous flow matching — decouples action token timing.
- Source: https://arxiv.org/abs/2511.14148

### FAST (Physical Intelligence)
- DCT-based action tokenization. Compresses high-freq continuous actions for AR LMs.
- Source: https://www.pi.website/download/fast.pdf

### OpenVLA
- Action tokens replace 256 least-used LLaMA tokens. Pure discrete AR.
- Source: https://openvla.github.io/

## 4. Streaming video tokenizers + AR video models

### MAGI-1 (Sand AI)
- Chunked AR video gen, monotonic denoise. Streams.
- Source: https://static.magi.world/static/files/MAGI_1.pdf

### CausVid
- Causal video diffusion. 1.3s initial latency, then ~9.4 FPS streaming on 1 GPU.

### Cosmos-Tokenizer (NVIDIA)
- Causal conv + causal temporal attention. Online streaming encode/decode.

### ProVideLLM
- Per-frame streaming inference at 10 FPS, dialogue at 25 FPS, 2GB VRAM. Sub-linear scaling.
- Source: https://arxiv.org/html/2504.13915v1

### AdapTok
- Adaptive per-sample token allocation, causal scorer.

### StreamAgent
- Anticipatory streaming video understanding. Hierarchical streaming KV-cache. Predicts future task-relevant intervals/regions.
- Source: https://arxiv.org/abs/2508.01875

## 5. World models — the other lens

### WebWorld (Qwen, 2026)
- 8B/14B/32B open-web world model. 1M+ web trajectories. Predicts next page state given action.
- Trains agents: +9.9% MiniWob++, +10.9% WebArena. As inference-time lookahead, beats GPT-5 as world model.
- Source: https://arxiv.org/html/2602.14721v1 (also HF: Qwen/WebWorld-8B)

### Genie 3 (DeepMind, 2026)
- 24 FPS persistent 3D interactive world. Real-time.
- Source: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/

### MobileDreamer
- Generative sketch world model for GUI agents.
- Source: https://arxiv.org/html/2601.04035v1

### Unified Video Action Model (Stanford)
- Joint video + action prediction in one AR backbone for robotics. Direct precedent for joint visual-motor AR.
- Source: https://unified-video-action-model.github.io/

## 6. The gap

Every existing computer-use agent is **half-duplex**:
- screenshot → tokens → CoT → action → execute → wait → next screenshot.
- Cycle 1-5s, blocks model during execution, cannot observe mid-action UI changes (animations, drag-trails, video, scroll inertia, hover states, autocomplete dropdowns).
- Action stream is discrete tool-calls (`click(x,y)`, `type("...")`). No continuous cursor motion. No timing precision.
- No world-model objective — the agent never predicts what the screen *will* look like. So no self-play, no dream rollouts, no native lookahead.

Speech-side has crossed the duplex chasm (Moshi → PersonaPlex → TML). GUI-side has not. The arch ingredients all exist; nobody has wired them together.
