# AI-Engineering-Awesome (ML Sys + Hardware/Software co-design)
*ai/ml resources to master state-of-the-art (SOTA) techniques from engineers and researchers* 🧠💻

---

## Foundational knowledge: End to end free guides to follow

<img height="50" alt="image" src="https://github.com/user-attachments/assets/82fdef14-cc94-4a78-bdd0-fd5e7d38bd0e" /> <img height="50" alt="image" src="https://github.com/user-attachments/assets/b7d30827-1b3d-4bb9-b792-8f47aa98e529" />

MUST:
- [ ] [AI Engineering handbook](https://amzn.to/3Wl5Tum): Book by DeepSeek covering **all major concepts in modern AI and AI engineering**; Must for reference (rating 10/10)
- [ ] [CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU): 20 videos on **ML basics** by Andrew Ng, Stanford University (rating 10/10)
- [ ] [CSE223](https://hao-ai-lab.github.io/cse234-w25/): **ML Sys** course by Prof Hao Zhang (rating 10/10) by UC San Diego (core engineering LLM serving concepts)
- [ ] [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=first_steps:_training_on_one_gpu): by HuggingFace on Training LLMs on GPU Clusters (rating 8.5/10)

---

Contents:
* Core AI engineering papers you MUST read
* Main AI blogs to read regularly (continuous learning)
* **Deep dive into all core AI concepts** [Learn step-by-step]
* MAYBE guides you may go through
* Want to contribute in leading AI open-source projects?

---

## Core AI engineering papers you MUST read

<img height="50" alt="image" src="https://github.com/user-attachments/assets/b0bcbfd3-5e89-4133-89a3-55e858fa82a5" /> <img height="50" alt="image" src="https://github.com/user-attachments/assets/50fbc127-b4b2-4bb9-8f02-cc72ad126da0" />

**Kernel level and memory system engineering**
- [ ] [AI and memory wall](https://arxiv.org/pdf/2403.14123): How memory is the main bottleneck for LLM?
- [ ] [Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs](https://arxiv.org/pdf/2512.18134) by NVIDIA and Stanford University
- [ ] [Collective Communication for 100k+ GPUs](https://arxiv.org/abs/2510.20171) by Meta
- [ ] [Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators](https://arxiv.org/pdf/2512.23236) by Meta
- [ ] [Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs](https://arxiv.org/pdf/2506.03296)
- [ ] [The Landscape of GPU-Centric Communication](https://arxiv.org/pdf/2409.09874v2)
- [ ] [Use microbenchmarking and instruction level analysis to analyze GPU architecture](https://arxiv.org/pdf/2208.11174)

**Algorithms for Kernel development**:
- [ ] [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- [ ] Flash Attention [paper1](https://arxiv.org/abs/2205.14135), paper for [v2](https://arxiv.org/abs/2307.08691), paper for [v3](https://tridao.me/blog/2024/flash3/)

**Training engineering**
- [ ] [Pre-training under infinite compute](https://arxiv.org/pdf/2509.14786) by Stanford University
- [ ] [Small Batch Size Training for Language Models](https://arxiv.org/pdf/2507.07101)
- [ ] [End-to-End Test-Time Training for Long Context](https://test-time-training.github.io/e2e.pdf) by NVIDIA and others
- [ ] [Give Me BF16 or Give Me Death](https://arxiv.org/pdf/2411.02355) by RedHat and [Give Me FP32 or Give Me Death?](https://arxiv.org/pdf/2506.09501v1)

**Others**:
- [ ] [LLMs don't just memorize, they build a geometric map that helps them reason](https://arxiv.org/pdf/2510.26745) by Google
- [ ] [Self-Adapting Language Models](https://arxiv.org/pdf/2506.10943) by MIT
---

## Main AI blogs to read regularly (continuous learning)

- [ ] [NVIDIA Developer Blog](https://developer.nvidia.com/blog/): Deep dive into multiple AI topics.
- [ ] [TensorRT LLM tech blogs](https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs/tech_blog): Deep dive into technical techniques/optimizations in one of the leading LLM inference library. (13 posts as of now)
- [ ] [SGLang tech blog](https://lmsys.org/blog/): SGLang is one of the leading LLM serving framework. Most blogs are around SGLang but is rich in technical information.
- [ ] [AI System co-design](https://aisystemcodesign.github.io/) at Meta

YouTube channels to follow regularly:

- [ ] [vLLM office hours](https://www.youtube.com/watch?v=uWQ489ONvng&list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3): Deep dive into various technical topics in vLLM
- [ ] [GPU Mode](https://www.youtube.com/@GPUMODE/videos): Deep dive into various LLM topics from guests from the AI community
- [ ] [PyTorch channel](https://www.youtube.com/@PyTorch/videos): videos of various PyTorch events covering keynotes of technical topics like torch.compile.
- [ ] [AI engineer channel](https://www.youtube.com/@aiDotEngineer/videos)

---

## Deep dive into AI concepts [Learn step-by-step]
_Listed only high-quality resources. No need to read 100s of posts to get an idea. Just one post should be enough._

* **GPU architecture**
<br> Current SOTA AI/LLM workloads are possible only because of GPUs. Understanding GPU architecture gives you an engineering edge.
- [ ] [Understanding GPU architecture with MatMul](https://www.aleksagordic.com/blog/matmul), [intuition about GPUs](https://jax-ml.github.io/scaling-book/gpus/)
- [ ] [GPU Shared memory banks / microbenchmarks](https://feldmann.nyc/blog/smem-microbenchmarks)
GPU programming concepts:
- [ ] [CUDA programming model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/), [GPU memory management](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62550/): Mark Harris's GTC Talk on Coalesced Memory Access, [Prefix Sum/ Scan in GPU](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [ ] [Programming Massively Parallel Processors](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) series on YT

* Performance
- [ ] [Performance metrics](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#performance-reported-by-nccl-tests) by nccl tests, [Profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#profiling-guide) by Nsight, [Understanding DL performance](https://horace.io/brrr_intro.html)

* **Transformer**
- [ ] [CME 295](https://cme295.stanford.edu/syllabus/): Basics of Transformer and LLM course by Stanford University
- [ ] [Transformer overall](https://www.krupadave.com/articles/everything-about-transformers): Encoder-only and Decoder-only models
- [ ] BERT (_insightful_): [BERT as text diffusion step](https://nathan.rs/posts/roberta-diffusion/)
- [ ] [Memory requirements for LLM](https://themlsurgeon.substack.com/p/the-memory-anatomy-of-large-language). There are 4 parts: activation, parameter, gradient, optimizer states. [How LLM handle memory?](https://fastpaca.com/blog/llm-memory-systems-explained)

* **Attention**

<img height="100" alt="image" src="https://github.com/user-attachments/assets/610b462d-ae36-4b25-a657-fd05f210eb53" /> <img height="100" alt="image" src="https://github.com/user-attachments/assets/39dabb64-bd91-40a0-870c-d1218ac005c3" />

- [ ] [Self-attention / Multi-head attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) (MHA), Multi-Query attention (MQA), [Group Query Attention](https://www.ibm.com/think/topics/grouped-query-attention) (GQA), MLA (used in DeepSeek)
- [ ] [FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) ([paper1](https://arxiv.org/abs/2205.14135), paper for [v2](https://arxiv.org/abs/2307.08691), paper for [v3](https://tridao.me/blog/2024/flash3/), Online softmax, [Implementation](https://github.com/Dao-AILab/flash-attention) by Tri Dao 
- [ ] [Ring Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-013/) (links to Context Parallelism CP): Handles large sequence length, [Flex Attention](https://arxiv.org/abs/2412.05496) by PyTorch
- [ ] KV cache, FP8 KV cache, [Paged Attention](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/)
- [ ] [Data Parallel (DP) Attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models)

**Core operations**

- [ ] [GEMM](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) / MatMul, [API of GEMM](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-0/gemm.html), [GEMM as core of AI](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/), [W4A8 GEMM Kernel](https://arxiv.org/pdf/2509.01229)
- [ ] [MoE](https://www.ibm.com/think/topics/mixture-of-experts) (Mixture of experts)
- [ ] [Embedding](https://huggingface.co/spaces/hesamation/primer-llm-embedding?section=bert_(bidirectional_encoder_representations_from_transformers)) (deepdive), RoPE ([paper](https://arxiv.org/pdf/2104.09864))

* **Quantization**

<img height="20" alt="image" src="https://github.com/user-attachments/assets/ad731cfe-ede3-4e53-b28b-e48221aab6c9" />

- [ ] [Neural Network Quantization guide](https://arxiv.org/pdf/2106.08295) by Qualcomm (cover PTQ, QAT)
- [ ] [Quantization basics](https://themlsurgeon.substack.com/p/the-machine-learning-surgeons-guide)
- [ ] [Different data type stimulations](https://www.quant.exposed/)
- [ ] [INT8 quantization using QAT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/), [LLM quantization with PTQ](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/), [FP8 datatype](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/), [AWQ](https://hamzaelshafie.bearblog.dev/awq-activation-aware-weight-quantisation/)
- [ ] [Per-tensor and per-block scaling](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [ ] [NVFP4 training](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/), [Optimizing FP4 Mixed-Precision Inference on AMD GPUs](https://lmsys.org/blog/2025-09-21-petit-amdgpu/), Recent [LLM quantization progress](https://blog.openvino.ai/blog-posts/q325-technology-update---low-precision-and-model-optimization)
- [ ] [Quantization on CPU (GGUF, AWQ, GPTQ)](https://www.ionio.ai/blog/llms-on-cpu-the-power-of-quantization-with-gguf-awq-gptq), [GGUF quantization method](https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/), [GPTQ](https://arxiv.org/pdf/2210.17323): Post training quantization for LLM. OBQ: [Post-Training Quantization and Pruning](https://arxiv.org/pdf/2208.11580)
- [ ] [Mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [ ] [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [GEMV kernel](https://veitner.bearblog.dev/nvfp4-gemv/) for NVFP4
- [ ] [Details on FP8 training](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/)
---
- [ ] [Pruning and distillation](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)

* Post-training
- [ ] [Post training concepts with SFT, RLHF, RLFR](https://tokens-for-thoughts.notion.site/post-training-101)
- [ ] [Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction)

* Optimizations

- [ ] [LLM Inference optimizations](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/); [optimizations v2](https://gaurigupta19.github.io/llms/distributed%20ml/optimization/2025/10/02/efficient-ml.html)
- [ ] 5D parallelism [PP, SP, DP, TP, CP, EP](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html), [parallelism](https://themlsurgeon.substack.com/p/data-parallelism-scaling-llm-training) concept for LLM scaling. [Parallelism in PyTorch](https://ggrigorev.me/posts/introduction-to-parallelism/)
- [ ] [Chunk prefill - SARATHI paper](https://arxiv.org/pdf/2308.16369), [dynamic and continuous batching](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching)
- [ ] [KV cache offloading](https://bentoml.com/llm/inference-optimization/kv-cache-offloading), [KVcache early reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)
- [ ] [Speculative decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding), ([basic introduction](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/speculative-decoding/speculative-decoding.md#speculative-decoding)), [Look-ahead reasoning](https://hao-ai-lab.github.io/blogs/lookaheadreasoning/), [Paper from Google](https://arxiv.org/pdf/2211.17192) and [DeepMind](https://arxiv.org/pdf/2302.01318)
- [ ] [MoE using Wide Expert Parallelism EP](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/)

<img height="100" alt="image" src="https://github.com/user-attachments/assets/68297949-8f6f-41e3-aa43-a9cae8c52102" />

* Scheduling / Routing

- [ ] [P/D disaggregation](https://hao-ai-lab.github.io/blogs/distserve-retro/), [DistServe P/D disaggregation paper](https://arxiv.org/pdf/2401.09670)
- [ ] [KVCache-centric disaggregated architecture](https://arxiv.org/pdf/2407.00079) by MooncakeAI
- [ ] [OverFill: Two-Stage Models for Efficient Language Model Decoding](https://arxiv.org/pdf/2508.08446) by Cornell University

* Software tools AI
- [ ] [vLLM arch](https://www.aleksagordic.com/blog/vllm): architecture of the leading LLM serving engine.

Insights:

- [ ] [MinMax M2 using Full Attention](https://x.com/zpysky1125/status/1983383094607347992): why full attention is better than masked attention?

Practical:

- [ ] [CUDA Compiler & PTX](https://blog.alpindale.net/posts/top_k_cuda/) with example
- [ ] [CUTLASS](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/): template library to code in CUDA easily
- [ ] [Matrix transpose using CUTLASS](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/)
- [ ] [SGLang inference engine architecture](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_sglang.pdf)
- [ ] [FlexAttention using CuTE DSL](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/)
- [ ] [MatMul using WGMMA](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/), [GEMM with pipelining in CUTLASS](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)


## MAYBE guides you may go through

- [ ] [Scaling a model](https://jax-ml.github.io/scaling-book/) by Jax (Google) (rating 7/10)
- [ ] [Smol training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction) by HuggingFace to train LLMs
- [ ] [GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3): if you want to dive deep into GPU programming
- [ ] (blog) [OpenVINO optimizations and engineering](https://blog.openvino.ai/) by Intel
- [ ] (blog) [Engineering posts by Colfax Research](https://research.colfax-intl.com/blog/)
- [ ] (blog) [GPU MODE lecture notes](https://christianjmills.com/series/notes/cuda-mode-notes.html)
- [ ] (blog) [Connectionism- Thinking Machine blog](https://thinkingmachines.ai/blog/): AI startup. Founded by Mira Murati, former CTO at OpenAI. Solved nondeterminism problem in LLM.
- [ ] [Llama visualization](https://www.alphaxiv.org/labs/tensor-trace): step by step [analyze each tensor](https://www.alphaxiv.org/labs/fly-through-llama) as it is processed in Llama

## Want to contribute in leading AI open-source projects?

Get started in these:

- [ ] [SGLang](https://github.com/sgl-project/sglang): LLM serving engine originally from UC Berkeley.
- [ ] [vLLM](https://github.com/vllm-project/vllm): LLM inference engine originally from UC Berkeley.
- [ ] [PyTorch](https://github.com/pytorch/pytorch): Leading AI framework by Meta
- [ ] [TensorFlow](https://github.com/tensorflow/tensorflow): AI framework by Google
- [ ] [TensorRT](https://github.com/NVIDIA/TensorRT): High performance inference library by NVIDIA
- [ ] [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): LLM inference library by NVIDIA
- [ ] [NCCL](https://github.com/NVIDIA/nccl): High performance GPU communication library by NVIDIA
- [ ] See other [NVIDIA libraries](https://github.com/orgs/NVIDIA/repositories?language=&q=&sort=&type=all).
