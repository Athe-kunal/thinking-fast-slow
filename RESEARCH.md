# Research log: thinking fast and slow with one tri-mode model

Running record of what we test, the assumptions behind each test, and what we
found. Newest experiment last within "Experiments". Every number here comes
from a run artifact under `runs/` (git-ignored, local to the machine) or from a
cited source; anything not verified is marked as such.

**Maintenance rule:** every new experiment gets an entry (question, setup,
assumptions, results, conclusion, artifacts) and any new global assumption goes
into the assumption register.

## 1. Idea

Use `nvidia/Nemotron-Labs-Diffusion-3B` as both a fast thinker (block
diffusion: many tokens per forward pass) and a slow thinker (autoregressive,
one token per pass), with a **router** that decides, from the model's causal
hidden state, which mode decodes the next segment. Diffusion can commit both
"fast thinking" tokens and action tokens; AR handles deliberate reasoning and
can validate diffusion output cheaply. Long term: train the router (RL on task
outcome + compute cost, possibly with a supervised warm start), backbone frozen.

## 2. Setup

| item | value |
|---|---|
| Model | `nvidia/Nemotron-Labs-Diffusion-3B` (3.83B params, bf16), Ministral3 backbone |
| Hardware | A100 80GB; **allowed GPUs: 3 (and 2 since 2026-10-03)**; shared machine |
| Driver | 550 (CUDA 12.4); main env torch 2.6.0+cu124, transformers 5.x |
| Model code | Local editable copy in `src/nemotron/` (HF remote code, unmodified) |
| Engine | `src/engine.py` modes `ar`, `dlm`, `mix`; `src/interleave.py` (routed decoding); `src/router.py` (routers) |
| Serving | `src/server.py` (OpenAI-compatible, mode via model name); SGLang fork in `.venv-sglang` (Sec. 5, E3) |

## 3. Verified facts about the architecture

- **One backbone, two attention patterns.** A single `encoder`
  (`Ministral3Model`) plus one LM head (`diffusion_head`). Mode = per-layer
  flag `self_attn.diffusion_lm` (True: bidirectional, False: causal).
- **The KV cache only ever holds causal state.** Diffusion denoising passes run
  with `use_cache=False`; a finished block is committed with one causal pass.
- **Switching on the causal hidden state is exact** (E2): at every switch point
  the router's hidden state matches a fresh causal pass over the same tokens
  (cosine 0.9998-1.0, same next token), and a committed diffusion block can be
  cut at any position k and continued in AR exactly.
- `ar_generate` / `generate` flip the flag in place; alternating them is safe
  (both set it before use).
- Fixed routers reproduce stock decoders token for token: `FixedRouter("ar")`
  == `ar_generate`, `FixedRouter("dlm")` == `generate(block 32, thr 0.9)`.

## 4. Assumption register (applies unless an experiment says otherwise)

| # | assumption | why | risk / how it could mislead |
|---|---|---|---|
| A1 | Greedy decoding everywhere (temperature 0) | Deterministic, matches HF examples | One sample per item; no sampling variance measured |
| A2 | Diffusion block 32, confidence threshold 0.9, first token seeded by the causal pass ("causal context") | HF chat-example defaults | Paper's diffusion setting uses block **8**; block size likely matters for code |
| A3 | Router decides per **segment**; in E4/E5 AR segments = 32 tokens so p = expected diffusion token share | Interpretable p | Coarse: switches only every 32 tokens; not how a real router would work (AR can hand over after any token) |
| A4 | Random router: one seed per (setting, benchmark, item) via sha256 | Resumable, worker-independent | Only one router draw per item; CIs cover item variance, not router-seed variance |
| A5 | Efficiency metric = tokens per forward pass (prefill = 1 pass, each denoising step = 1, each block commit = 1) | Hardware-independent | Treats a 32-token pass as costing the same as a 1-token pass; not wall-clock. Close to, not identical with, NVIDIA's TPF counting |
| A6 | Chat template defaults: `enable_thinking=False` (thinking off) | Template default; NVIDIA `evaluate.py` doesn't set it | Paper's numbers may use thinking (unknown) |
| A7 | Long prompts prefilled in 8192-token chunks, logits only at the last position | Memory (fix for OOM at 66k-token contexts) | Verified identical outputs on a 17k-token prompt |
| A8 | Answer budget 1024 tokens | Throughput | Paper uses 8192; only 0-2 truncations per setting in E5 |
| A9 | tok/s numbers are **not** speed measurements when several workers share a GPU | - | Use tokens/forward for efficiency |

## 5. Experiments

### E1. Harbor toy agent tasks (2026-10-02)

- **Question:** does the full agent pipeline (server + mini-swe-agent) run in AR and diffusion modes?
- **Setup:** 2 self-written Harbor-format tasks (`tasks/hello-world`, `tasks/sum-numbers`), mini-swe-agent text-based config, step limit 15, hub remote-code model, diffusion block 32 / thr 0.9.
- **Results:**

  | task | mode | reward | exit | steps |
  |---|---|---|---|---|
  | hello-world | ar | 1 | Submitted | 2 |
  | hello-world | dlm | 1 | LimitsExceeded (never submitted) | 15 |
  | sum-numbers | ar | 1 | Submitted | 2 |
  | sum-numbers | dlm | 0 | LimitsExceeded (looped on relative path `app/sum.py`) | 15 |
- **Conclusion:** pipeline works; n=2, no statistical weight.
- **Artifacts:** `runs/*.traj.json`.

### E2. Can we switch modes from the last causal hidden state? (2026-10-03)

- **Setup:** `scripts/check_switch.py`, GPU 3, CycleRouter (ar8 dlm32 ...), 96 tokens; mid-block cut at k=11 then 16 AR tokens.
- **Results:** switch-point cosine vs fresh causal pass 0.99982-1.00000, same next token at all 6 switch points; mid-block positions 3/11/31 cosine 0.99976-0.99988; crop-then-AR == `ar_generate` from same prefix: **True**. Differences are bf16 incremental-vs-full-prefill noise.
- **Conclusion:** routing on the causal hidden state is exact; diffusion blocks can be cut at any position after the causal commit; no switching mid-denoising.

### E3. SGLang DLLM fork veracity (2026-10-03)

- **Setup:** fork `hutm/sglang` @ `23472a0` (branch `upstream/2-dllm-linearspec`; NVIDIA's guide branch `upstream/2-dllm-lora-ar` no longer exists); upstream PR sgl-project/sglang#25803 open, changes requested. CUDA 13 pins swapped for CUDA 12.x builds (`configs/sglang/overrides.txt`). `scripts/compare_sglang.py`: 30 GSM8K questions, own prompt with `####`, 512 tokens, greedy, concurrency 1, GPU 3.
- **Results:**

  | mode | SGLang acc | HF acc | identical outputs | SGLang tok/s | HF tok/s |
  |---|---|---|---|---|---|
  | linear_spec | 86.7% | 86.7% | 19/30 | 450 | 187 |
  | dlm (FastDiffuser, thr 0.9) | 83.3% | 86.7% | 8/30 | 291 | 116 |
  | ar (`ar_mode` + FastDiffuser) | **36.7%** | 86.7% | 0/30 | 109 | 51 |
- **Conclusion:** SGLang LinearSpec is trustworthy and ~2.4x faster; SGLang "AR mode" is broken (diffusion loop over causal attention, only tested for non-empty output upstream). One decoding mode per server; no router support.

### E4. SWE-bench Verified with a random router (2026-10-03, stopped early)

- **Question:** how does an untrained (random) router do on real SWE tasks at p(diffusion) = 0.2 vs 0.8?
- **Setup:** `scripts/swebench_router_experiment.py`; 50 instances sampled from Verified (seed 0; `runs/swebench_random_router/instances.json`); mini-swe-agent `swebench_backticks.yaml` (text actions); step limit 50, 1024 tokens/step, 60 s per command, no context truncation; `mix` mode, RandomRouter, AR segments 32; 2 lanes x 2 settings = 4 servers on GPU 3; scoring with `swebench` 4.1.0 harness (validated: gold patch resolves, test-only patch does not).
- **Results (stopped after ~19 runs per setting):**

  | | p=0.2 | p=0.8 |
  |---|---|---|
  | finished / scored | 18 / 10 | 19 / 10 |
  | resolved | 0 | 0 |
  | non-empty patch | 0/18 | 0/19 |
  | exit statuses | LimitsExceeded 15, Submitted (empty) 3 | LimitsExceeded 15, Submitted (empty) 2, RepeatedFormatError 2 |
  | mean steps | 43.6 | 45.3 |
  | format errors / step | 0.03 | 0.06 |
  | diffusion token share | 18.9% | 78.9% |
  | tokens / forward | 1.16 | 2.30 |
- **Trace analysis (18 runs each):** opened the file the gold fix changes 0/18 vs 2/18; ran tests 2/18 vs 2/18; repeated one command 5+ times 16/18 vs 16/18. Typical failures: blind repo-wide `sed` with no reading, find/sed loops, copying the prompt's placeholder `git diff -- path/to/file1 ...`.
- **Conclusion:** the 3B model is at a floor of zero on SWE-bench Verified in both settings, so this task carries **no signal about routing**. Pure AR (p=0) was not run.
- **Issues found and fixed during E4:** (1) prefill computed full-prompt logits (~10 GB at 40k tokens) -> last-position only; (2) OOM at 66k-token contexts -> chunked prefill (A7) + per-server memory cap; (3) harness summary file missing after cleanup crash -> read per-instance `report.json`. Run restarted from scratch after each code change.
- **Artifacts:** `runs/swebench_random_router/` (`summary.json`, per-lane trajectories and request logs).

### E5. Random router on GSM8K and HumanEval (2026-10-03)

- **Question:** does random AR/diffusion routing keep accuracy, and how does it trade accuracy for efficiency?
- **Setup:** `scripts/eval_router_benchmarks.py`; full GSM8K test (1319) and HumanEval (164); p in {0, 0.2, 0.8, 1}; segments 32/32 (A3); 1024 tokens; GSM8K prompt and scoring copied from NVIDIA `evaluate.py`; HumanEval: own prompt (full function in one code block), first code block, stdlib-only network-less Docker sandbox, 10 s timeout. 12 workers on GPUs 2 and 3 (restart from 6 workers on GPU 3 kept finished records).
- **Results:**

  | benchmark | p | accuracy | 95% CI | diffusion share | tokens/forward | mean length |
  |---|---|---|---|---|---|---|
  | GSM8K | 0 (AR) | 87.3% | 85.4-89.0 | 0% | 1.00 | 177 |
  | GSM8K | 0.2 | 87.5% | 85.6-89.2 | 19.8% | 1.14 | 174 |
  | GSM8K | 0.8 | 87.8% | 85.9-89.5 | 79.8% | 2.01 | 174 |
  | GSM8K | 1 (diffusion) | 87.3% | 85.4-89.0 | 100% | 2.72 | 173 |
  | HumanEval | 0 (AR) | 80.5% | 73.8-85.8 | 0% | 1.00 | 210 |
  | HumanEval | 0.2 | 79.3% | 72.4-84.8 | 19.7% | 1.16 | 273 |
  | HumanEval | 0.8 | 69.5% | 62.1-76.0 | 79.7% | 1.96 | 374 |
  | HumanEval | 1 (diffusion) | 72.6% | 65.3-78.8 | 100% | 2.46 | 415 |

  Paired exact McNemar tests (items solved by only one setting):
  - HumanEval: AR vs 0.8: 21 vs 3, **p<0.001**; AR vs 1.0: 20 vs 7, **p=0.019**; AR vs 0.2: 8 vs 6, p=0.79; 0.8 vs 1.0: 4 vs 9, p=0.27.
  - GSM8K: all pairs p >= 0.44.
- **Checks:** HumanEval drop is not an extraction artifact (at most 1/164 outputs per setting have more than one code block; inspected failures are genuinely wrong code). Longer diffusion outputs come from added explanation text. Our AR/diffusion GSM8K (87.3%) matches NVIDIA's 3B table (87.87% AR, 88.40% Diff.).
- **Conclusion:** switching never breaks generation. On math, routing is free (2-2.7x fewer forward passes, no accuracy change). On code, diffusion costs ~8-11 points, so a learned router has real headroom there: target AR's ~80% at diffusion-level efficiency.
- **Caveats:** A2 (block 32 vs paper's 8), A4 (one router draw), own HumanEval prompt (absolute numbers not comparable to NeMo-Skills).
- **Artifacts:** `runs/router_bench/` (`summary.json`, `scored.jsonl`, `worker*.jsonl`).

### E6. Entropy router (0.5 nats) on GSM8K and HumanEval (2026-10-03)

- **Question:** does routing on next-token entropy beat random routing at a similar diffusion share?
- **Setup:** identical to E5 (same prompts, scoring, block 32 / thr 0.9, 32-token AR segments, 1024 tokens, greedy), router `entropy:0.5`: at each segment boundary, compute the entropy (nats, float32, full vocabulary) of the causal next-token distribution; diffusion if < 0.5, else AR. Boundaries always follow a causal pass (AR token or block commit). 12 workers on GPUs 2 and 3; records written next to E5's in `runs/router_bench/`. Deterministic router (no seed).
- **Results:**

  | benchmark | router | accuracy | 95% CI | diffusion share | tokens/forward | mean length |
  |---|---|---|---|---|---|---|
  | GSM8K | entropy:0.5 | 87.7% | 85.8-89.4 | 89.8% | 2.34 | 173 |
  | GSM8K | random:0.8 (E5) | 87.8% | 85.9-89.5 | 79.8% | 2.01 | 174 |
  | GSM8K | random:1.0 (E5) | 87.3% | 85.4-89.0 | 100% | 2.72 | 173 |
  | HumanEval | entropy:0.5 | **76.2%** | 69.2-82.1 | 87.5% | 2.17 | 420 |
  | HumanEval | random:0.0 = AR (E5) | 80.5% | 73.8-85.8 | 0% | 1.00 | 210 |
  | HumanEval | random:0.8 (E5) | 69.5% | 62.1-76.0 | 79.7% | 1.96 | 374 |
  | HumanEval | random:1.0 (E5) | 72.6% | 65.3-78.8 | 100% | 2.46 | 415 |

  Paired exact McNemar (only-A vs only-B solved):
  - HumanEval: entropy vs AR 11 vs 18, p=0.27; entropy vs random:0.8 16 vs 5, **p=0.027**; entropy vs random:1.0 10 vs 4, p=0.18.
  - GSM8K: entropy vs AR p=0.73; vs random:0.8 p=1.0; vs random:1.0 p=0.45.
- **Conclusion (revised after E7):** on HumanEval the entropy router beat random:0.8 (+6.7 points, p=0.027) while using more diffusion, but E7 shows random:0.8 was an unusually low draw (random:0.9 scored 73.8%). Against the matched-share random:0.9 the gain is +2.4 points, not significant (p=0.45). Not distinguishable from pure AR (-4.3, p=0.27) or pure diffusion (+3.6, p=0.18) at n=164. GSM8K: no differences, as in E5.
- **Caveats:** 0.5 nats picks diffusion ~88-90% of the time, so this is one point on the curve, not a matched-share comparison (no random run at ~88%); random routers have one draw per item (A4); E5/E6 block-32 setting (A2); outputs stay long (420 tokens) like diffusion.
- **Artifacts:** `runs/router_bench/` (`setting: "entropy:0.5"` records; `summary.json`, `scored.jsonl`).

### E7. Entropy threshold sweep vs random routing on HumanEval (2026-10-03)

- **Question:** what accuracy/efficiency curve do entropy routers trace, and do they beat random routing at matched diffusion share?
- **Setup:** identical to E5/E6 (HumanEval only); new settings entropy:{0.05, 0.1, 0.25, 1.0, 2.0} and random:{0.5, 0.9}; 12 workers on GPUs 2 and 3.
- **Results (HumanEval, n=164):**

  | router | accuracy | diffusion share | tokens/forward | mean length |
  |---|---|---|---|---|
  | random:0.0 (AR) | 80.5% | 0% | 1.00 | 210 |
  | random:0.2 | 79.3% | 19.7% | 1.16 | 273 |
  | random:0.5 | 76.2% | 47.2% | 1.44 | 317 |
  | entropy:0.05 | 75.6% | 74.3% | 1.91 | 419 |
  | entropy:0.1 | 75.0% | 78.3% | 1.98 | 422 |
  | random:0.8 | 69.5% | 79.7% | 1.96 | 374 |
  | entropy:0.25 | 75.6% | 82.6% | 2.07 | 422 |
  | entropy:0.5 | 76.2% | 87.5% | 2.17 | 420 |
  | random:0.9 | 73.8% | 90.6% | 2.22 | 395 |
  | entropy:1.0 | 75.6% | 97.0% | 2.38 | 414 |
  | entropy:2.0 | 72.6% | 99.9% | 2.45 | 416 |
  | random:1.0 (diffusion) | 72.6% | 100% | 2.46 | 415 |

  Matched-share paired McNemar: entropy:0.5 vs random:0.9 10 vs 6, p=0.45; entropy:1.0 vs random:1.0 6 vs 1, p=0.13; entropy:0.1 vs random:0.8 16 vs 7, p=0.09; random:0.9 vs random:0.8 11 vs 4, p=0.12. Mean accuracy entropy 0.05-1.0 = 75.6% vs random 0.8-1.0 = 72.0%.
- **Findings:**
  1. **Entropy at segment boundaries is almost always tiny.** Even a 0.05-nat threshold sends 74% of tokens to diffusion, so this router cannot reach the low-diffusion part of the curve where accuracy is near AR's.
  2. **Flat plateau at ~75-76%** for thresholds 0.05-1.0 (74-97% diffusion), then a drop to pure-diffusion level at 2.0. The gap to AR (~4-5 points) does not close with any threshold.
  3. **Entropy is consistently a little above random at similar share** (+2.4 to +3.6 points at the matched points, +3.6 on average), and every entropy point at >= 74% share sits above the random points at 80-100%, but no single matched comparison is significant at n=164.
  4. **Random routing is noisy** (random:0.8 69.5% < random:0.9 73.8%): one router draw per item (A4) can move results by ~4 points, which inflated E6's significance.
  5. Outputs stay long (~420 tokens) whenever diffusion share is high, even with the entropy router.
- **Conclusion:** next-token entropy at segment boundaries is a weak but plausibly real signal (small, consistent edge over random) with too little dynamic range to keep code accuracy near AR. A better signal or finer granularity is needed (entropy over a window or a diffusion draft; AR decisions every 1-8 tokens; smaller diffusion blocks).
- **Artifacts:** `runs/router_bench/` (`summary.json`, `scored.jsonl`).

### E8. Per-token entropy check while in AR (2026-10-04)

- **Question:** if the router checks entropy after *every* AR token and switches to diffusion as soon as it drops below the threshold, does it keep AR's accuracy at higher efficiency?
- **Setup:** as E5-E7 but AR segments of 1 token (setting suffix `@ar1`): after each AR token, next-token entropy < threshold -> one 32-token diffusion block, else one more AR token. Diffusion blocks unchanged (32 tokens, committed whole; the next check happens after the block's causal commit). Settings: entropy:0.5@ar1 on GSM8K + HumanEval, entropy:0.05@ar1 on HumanEval. 12 workers on GPUs 2 and 3.
- **Results:**

  | benchmark | router | accuracy | 95% CI | diffusion share | tokens/forward |
  |---|---|---|---|---|---|
  | GSM8K | entropy:0.5@ar1 | 87.6% | 85.8-89.3 | 99.5% | 2.68 |
  | HumanEval | entropy:0.5@ar1 | 73.8% | 66.6-79.9 | 99.4% | 2.44 |
  | HumanEval | entropy:0.05@ar1 | 73.8% | 66.6-79.9 | 98.3% | 2.46 |
  | HumanEval | entropy:0.5 (32-token AR, E6) | 76.2% | 69.2-82.1 | 87.5% | 2.17 |
  | HumanEval | pure diffusion (E5) | 72.6% | 65.3-78.8 | 100% | 2.46 |
  | HumanEval | pure AR (E5) | 80.5% | 73.8-85.8 | 0% | 1.00 |

  AR usage (HumanEval): 0.5 nats -> 1.7 AR runs per answer, mean run 1.4 tokens (max 6); 0.05 nats -> 3.7 runs, mean 1.9 tokens (max 12).
  Paired McNemar: entropy:0.5@ar1 vs pure diffusion 5 vs 3, p=0.73; vs AR 9 vs 20, p=0.061; vs entropy:0.5 (32-token AR) 6 vs 10, p=0.45. GSM8K vs pure diffusion p=0.49.
- **Findings:**
  1. **Per-token checking makes the router almost pure diffusion.** Entropy rises above the threshold for one token, AR decodes that token, entropy falls, and diffusion takes the next 32-token block. Even 0.05 nats leaves AR only ~2% of tokens.
  2. **Handing AR the single uncertain tokens at block boundaries does not help:** accuracy equals pure diffusion (73.8% vs 72.6%, p=0.73). The efficiency equals pure diffusion too.
  3. **The asymmetry is the problem:** the router can leave AR after any token, but once in diffusion it commits 32 tokens with no check, and the uncertain tokens inside a block are decided by diffusion itself. Boundary entropy (right after a commit) is almost always low.
- **Conclusion:** the decisions that matter are inside diffusion blocks. Next idea: after a block's causal commit pass (which already yields AR logits for every block position, E2), cut the block at the first position whose causal entropy (or AR disagreement) is high and continue in AR from there; or use smaller diffusion blocks (8).
- **Artifacts:** `runs/router_bench/` (`setting` ends in `@ar1`).

### E9. Entropy router with 8-token AR segments (2026-10-04)

- **Question:** between E6 (AR commits 32 tokens per decision) and E8 (1 token), does committing 8 AR tokens per entropy decision do better?
- **Setup:** as E6/E8 with setting `entropy:0.5@ar8`: when next-token entropy >= 0.5 nats, decode 8 AR tokens, then check again; below 0.5, one 32-token diffusion block. GSM8K + HumanEval, 12 workers on GPUs 2 and 3.
- **Results (all entropy routers at 0.5 nats, by AR segment length):**

  | benchmark | AR segment | accuracy | 95% CI | diffusion share | tokens/forward |
  |---|---|---|---|---|---|
  | HumanEval | 32 (E6) | 76.2% | 69.2-82.1 | 87.5% | 2.17 |
  | HumanEval | **8** | 73.8% | 66.6-79.9 | 96.0% | 2.38 |
  | HumanEval | 1 (E8) | 73.8% | 66.6-79.9 | 99.4% | 2.44 |
  | HumanEval | pure diffusion | 72.6% | 65.3-78.8 | 100% | 2.46 |
  | HumanEval | pure AR | 80.5% | 73.8-85.8 | 0% | 1.00 |
  | GSM8K | 32 (E6) | 87.7% | 85.8-89.4 | 89.8% | 2.34 |
  | GSM8K | **8** | 87.9% | 86.1-89.6 | 96.8% | 2.60 |
  | GSM8K | 1 (E8) | 87.6% | 85.8-89.3 | 99.5% | 2.68 |

  AR usage with 8-token segments (HumanEval): 1.6 AR runs per answer, mean 10.2 AR tokens per run.
  Paired McNemar (HumanEval): ar8 vs pure diffusion 7 vs 5, p=0.77; vs ar32 4 vs 8, p=0.39; vs ar1 7 vs 7, p=1.0; vs AR 12 vs 23, p=0.09. GSM8K: all p >= 0.22.
- **Findings:** shorter AR segments monotonically shift the router toward diffusion (87.5% -> 96.0% -> 99.4% share) and toward pure-diffusion accuracy on HumanEval (76.2% -> 73.8% -> 73.8%, vs 72.6% pure diffusion). The extra AR tokens in the 32-token setting come from the router being *forced* to keep decoding AR, not from it choosing AR. GSM8K is unaffected.
- **Conclusion:** with boundary next-token entropy as the signal, the AR segment length mostly sets how much AR is forced, i.e. a crude knob on the diffusion share; it does not make the router smarter. Confirms E8: the useful decisions must be made inside diffusion blocks.
- **Artifacts:** `runs/router_bench/` (`setting: "entropy:0.5@ar8"`).

## 6. Reference numbers and related work

- **NVIDIA 3B table (paper, not reproduced by us except GSM8K):** AR avg 55.50 @ TPF 1.00; Diff. 52.90 @ 1.91; Linear SS 55.00 @ 4.36; Quad SS 55.80 @ 5.42. HumanEval AR 76.22 / Diff. 74.39 / Linear SS 75.00. Quad SS is not in the released code.
- **Training data (tech report):** Ministral3 base -> 1T tokens pure AR + 300B joint AR+diffusion (alpha=0.3) on the Nemotron Nano 2 pretraining set; SFT 45B tokens from the Nemotron 3 Super SFT set (joint loss, alpha=0.3). Possible SWE/agentic contamination unverified.
- **LinearSpec** (in released code): fixed rule, draft block with diffusion, verify causally, accept matching prefix + bonus token. Not a learned router.
- **LearnedSampler** (staged SGLang branch, unmerged, no public checkpoint): small transformer scoring which masked tokens to commit inside diffusion; inputs are top-k probs/entropy/token features, not the backbone hidden state.
- **S2D2** (arXiv 2603.25702): training-free self-speculation for block diffusion; routing policies decide when to verify (min-span, score-threshold, hysteresis, UCB bandit); bandit was not the best policy; entropy-based estimator gave the best accuracy. Tested on SDAR, Fast-dLLM v2, LLaDA2.1, not Nemotron.
- **SDAR:** AR-initialised block-diffusion model; single decoding mode, fixed block per checkpoint; no inference-time switching.

## 7. Open questions and next steps

1. Does entropy predict where diffusion fails? Offline check: at each block start in AR outputs, compare candidate signals (next-token entropy, rolling entropy over recent AR tokens, mean entropy over a diffusion draft) with the draft's match length vs AR. E7: boundary next-token entropy is weak and saturates.
2. Decide inside diffusion blocks (E8): use the commit pass's per-position causal entropy / AR disagreement to cut a block and hand over to AR; also diffusion block 8. (Per-token AR checks alone, E8, just give pure diffusion.)
3. Multiple router seeds for random baselines (A4) so the random curve is smooth enough to compare against; report measured diffusion share instead of p.
4. Oracle headroom vs LinearSpec (4.36 TPF) before investing in RL.
