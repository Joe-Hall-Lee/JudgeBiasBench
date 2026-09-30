# Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization

<div align="center">

![JudgeBiasBench](https://github.com/Joe-Hall-Lee/JudgeBiasBench/blob/main/assets/logo.png)

</div>

This is the official repository for [Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization](https://arxiv.org/abs/2603.08091).

It contains:

- **JudgeBiasBench** — a benchmark of 12 bias types in a four-category taxonomy (superficial quality, context, presentation, diversity) for judgement bias evaluation;
- **evaluation code** for generative and discriminative judges, plus four general judge benchmarks (RewardBench, JudgeBench, RM-Bench, RMB);
- **training code** for our debiased judges: a contrastive (InfoNCE) discriminative judge, and a generative judge trained with SFT cold-start + GRPO.

---



## Repository layout

```text
JudgeBiasBench/
├─ data/
│  ├─ eval/                      # JudgeBiasBench (12 x *_bias.jsonl) + reward_bench/ general benchmarks
│  └─ train/                     # training data, see "Training data" below
├─ common/api.py                 # shared OpenAI-compatible client (API_KEY / API_BASE_URL env vars)
├─ construct/                    # building the benchmark (HelpSteer3 sampling + bias injection) & the bias-augmented training negatives
├─ evaluate/
│  ├─ eval_judge.py              # generative judges via vLLM on the 12 bias sets (+ rewardbench/rmbench/judgebench/rmb)
│  ├─ eval_api.py                # same, for API models (GPT / Claude / Gemini / DeepSeek ...)
│  ├─ eval_general_benchmarks.py # generative judges on RewardBench / JudgeBench / RM-Bench / RMB with position-swap protocol
│  ├─ eval_rm.py                 # discriminative judges on the 9 response-level bias sets
│  ├─ eval_rm_{rewardbench,judgebench,rmbench,rmb}.py
│  ├─ *_handler.py, build_prompt.py, prompts/, model_utils.py, rm_utils.py, inference_module.py
│  └─ configs/                   # vLLM configs for eval_general_benchmarks.py
├─ train/
│  ├─ discriminative/            # contrastive / pairwise discriminative judge training (Accelerate + DeepSpeed), data_prep/
│  └─ generative/                # sft/ (LLaMA-Factory yamls), grpo/ (EasyR1 launcher + reward functions), process_data/,
│                                #   LLaMA-Factory/ and EasyR1/ (vendored copies of the two frameworks)
├─ scripts/                      # job scripts for the paper experiments (generative/, discriminative/) + shared lib/, evaluation wrappers, analyze_bsr_stats.py
├─ requirements.txt              # evaluation + discriminative judge training
├─ requirements-sft.txt          # SFT (LLaMA-Factory)
└─ requirements-grpo.txt         # GRPO (EasyR1)
```



## Setup

```bash
git clone https://github.com/Joe-Hall-Lee/JudgeBiasBench.git
cd JudgeBiasBench
```

The three stacks have conflicting dependencies (different vLLM / transformers versions), so we use three conda
environments. Only the first one is needed to evaluate judges on the benchmark.

```bash
# 1) evaluation + discriminative judge training
conda create -n jbb-eval python=3.12 && conda activate jbb-eval
pip install -r requirements.txt                       # vllm 0.9.1, torch 2.7.0, transformers 4.51.3

# 2) generative judge, SFT cold-start
conda create -n jbb-sft python=3.11 && conda activate jbb-sft
pip install -r requirements-sft.txt                   # LLaMA-Factory + torch 2.7.0 / transformers 4.51.3

# 3) generative judge, GRPO
conda create -n jbb-grpo python=3.11 && conda activate jbb-grpo
pip install -r requirements-grpo.txt                  # vllm 0.11.0, torch 2.8.0, transformers 4.56.2
pip install --no-deps --no-build-isolation -e ./train/generative/EasyR1
pip install flash-attn --no-build-isolation           # needs nvcc; a prebuilt wheel also works
```

The experiment scripts in `scripts/` switch between these environments by name (override with `ENV_EVAL` / `ENV_SFT` / `ENV_GRPO`).

API access (data construction, `eval_api.py`) is configured through environment variables, or a local `.env` file that stays out of the repo:

```bash
export API_KEY=sk-...                  # required
export API_BASE_URL=https://api.openai.com/v1   # optional, any OpenAI-compatible endpoint
```



## Data

`data/eval/<bias>_bias.jsonl` holds the 12 bias test sets. Field layout by how the bias is injected:


| Family                                       | Biases                                                                        | Fields                                                         |
| -------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Rewrite (superficial quality)                | length, authority, beauty, assertiveness, sycophancy, sentiment, concreteness | `question, original_response1/2, rewritten_response1/2, label` |
| Identity injection (diversity)               | gender, race                                                                  | same as rewrite (`rewritten_response*` carry the identity prefix) |
| Layout manipulation (context / presentation) | bandwagon, superficial-reflection, position                                   | `question, response1, response2, label`                        |


`data/eval/reward_bench/` holds local copies of RewardBench, JudgeBench, RM-Bench, RMB (pairwise), EvalBiasBench and IFBench in a unified `prompt / chosen / rejected` format.

## Evaluation

### Generative judges (vLLM)

```bash
python evaluate/eval_judge.py all \
  --model_path /path/to/model --model_name my-judge --model_type default \
  --temperature 0.0 --max_new_token 2048 --tensor_parallel_size 1 --gpu_memory_utilization 0.9
```

- `bias_type`: one of the 12 biases, `rewardbench|rmbench|judgebench|rmb`, or `all` (the 12 biases).
- `--model_type` selects the prompt format: `default` for general-purpose LLMs (short explanation, then `[[A]]`/`[[B]]`), the native formats of the fine-tuned judge baselines `judgelm|auto-j|selene|prometheus`, and `think` / `direct` — the `<think>…</think>` CoT prompt and the label-only prompt used to train and evaluate our SFT+GRPO judges. Templates are in `evaluate/build_prompt.py` and `evaluate/prompts/`.
- `--enable_thinking` toggles Qwen3-style thinking in the chat template.
- Outputs go to `results/<model_name>/` by default (`--output_dir` or `JBB_RESULTS_DIR` to change): per-sample `<bias>_<model_name>.jsonl`, per-bias `<bias>_summary_<model_name>.json`, and `all_biases_summary_<model_name>.json` with aggregated **Acc_ori / Acc_inj / BSR** (bias sensitivity rate). Re-running `all` skips biases that already have a summary (`--overwrite` to redo).

`scripts/run_evaluation_judge.sh` wraps the above with environment variables (`MODEL_PATH`, `MODEL_NAME`, `MODEL_TYPE`, `GPU_NUM`, `OUTPUT_DIR`).

### API models

```bash
API_KEY=... python evaluate/eval_api.py all --model gpt-4o
```



### General benchmarks for generative judges

```bash
python evaluate/eval_general_benchmarks.py --config evaluate/configs/Qwen3-8B.yaml --name qwen3-8b \
  --benchmarks rewardbench,judgebench,rm-bench,rmb_pairwise [--model_path /override/model]
```

`prompt:` in the config is `pair_cot` / `pair_no_cot` (our judges, general LLMs) or one of the `--model_type` names above
for the fine-tuned judge baselines, e.g. `selene`. Each pair is judged in both orders (position swap); a pair counts as correct only if both orders agree with the label. Results: `results/<name>/<bench>.json` and `benchmark_summary.json`.

### Discriminative judges

Works with our `RewardModel` checkpoints (`reward_head.pt` next to the backbone) and with any HF `AutoModelForSequenceClassification` model.

```bash
python evaluate/eval_rm.py all --model_path /path/to/judge --model_name my-judge --batch_size 32
python evaluate/eval_rm_rewardbench.py all --model_path /path/to/judge --model_name my-judge
python evaluate/eval_rm_judgebench.py  all --model_path /path/to/judge --model_name my-judge
python evaluate/eval_rm_rmbench.py     all --model_path /path/to/judge --model_name my-judge
python evaluate/eval_rm_rmb.py         all --model_path /path/to/judge --model_name my-judge
# or: MODEL_PATH=/path/to/judge WITH_BENCHMARKS=1 bash scripts/run_evaluation_rm.sh
```



## Training

Base model for all our judges: Qwen2.5-7B-Instruct.

Each training run of the paper is one resumable job script that trains, then evaluates on JudgeBiasBench and the
general benchmarks; re-running the same command continues from the latest checkpoint:

```bash
bash scripts/generative/sft_cot_grpo.sh                  # SFT (GPT-4o CoT) -> GRPO         [main generative judge]
bash scripts/generative/grpo_only.sh                     # GRPO without SFT cold-start
bash scripts/generative/sft_no_cot_grpo.sh               # SFT without teacher reasoning -> GRPO
bash scripts/generative/sft_grpo_no_format_reward.sh     # SFT -> GRPO without format reward
bash scripts/discriminative/contrast_both_negatives.sh   # InfoNCE, original + bias-augmented negatives [main discriminative judge]
bash scripts/discriminative/contrast_original_negatives.sh
bash scripts/discriminative/hinge_bias_negatives.sh
bash scripts/discriminative/hinge_both_negatives.sh
```

Common overrides: `BASE_MODEL` (local dir or hub id), `CKPT_ROOT`, `STAGE=train|eval`, `N_GPU` / `CUDA_VISIBLE_DEVICES`,
`FORCE_RETRAIN=1`, `SETUP_ENV=0` (already inside the right env), `JBB_TRAIN_DATA_DIR`; generative runs also take `STEP`
(GRPO step to evaluate, default 1800), `GRPO_DATA_DIR` and `GRPO_EXTRA_ARGS`, discriminative runs `DS_CONFIG`. The two
sections below show the underlying commands.

### Training data

All training data lives under `data/train/` (relocate with `JBB_TRAIN_DATA_DIR`) and is rebuilt from
[GRAM-fine-tuning-65K](https://huggingface.co/datasets/NiuTrans/GRAM-fine-tuning-65k) preference pairs.

**Discriminative judge** — `data/train/discriminative/*.jsonl`, one JSON object per line:

```json
{
  "question": "...",
  "chosen": "...",
  "rejected": ["original rejected response", "bias-augmented rejected response"],
  "bias_type": ["length"],
  "verify_correctness": "CHOSEN_IS_BEST"
}
```

Only rows with `verify_correctness == "CHOSEN_IS_BEST"` are used.

### Discriminative judge

`train/discriminative/train_reward_model.py` puts a scalar reward head on a causal LM (saved as backbone weights +
`reward_head.pt`, loaded by `evaluate/rm_utils.py`). Loader keeps only rows with `verify_correctness == "CHOSEN_IS_BEST"`.

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct OUTPUT_PATH=/path/to/ckpts/rm_contrast TRAIN_MODE=contrast \
bash train/discriminative/run_training.sh
```

- `TRAIN_MODE=contrast`: InfoNCE over the chosen response vs. all rejected responses of a row.
- `TRAIN_MODE=pairwise`: Hinge loss over chosen vs. `rejected[0]` only.

### Generative judge (SFT cold-start + GRPO)

LLaMA-Factory and EasyR1 are vendored under `train/generative/` so the exact versions are pinned. Run from the repo root.

```bash
# SFT on GPT-4o <think> traces (jbb-sft env); qwen2_5_7b_sft_no_cot.yaml = label-only ablation
FORCE_TORCHRUN=1 llamafactory-cli train train/generative/sft/qwen2_5_7b_sft_cot_gpt4o.yaml \
    model_name_or_path=/path/to/Qwen2.5-7B-Instruct output_dir=/path/to/ckpts/sft_cot_gpt4o

# GRPO (jbb-grpo env): reward = 0.9 x [[A]]/[[B]] accuracy + 0.1 x format; REWARD_FN=.../bias_no_format.py drops the format term
MODEL_PATH=/path/to/ckpts/sft_cot_gpt4o SAVE_PATH=/path/to/ckpts/sft_cot_grpo bash train/generative/grpo/run_grpo.sh

# merge the FSDP checkpoint to HF format, then evaluate with evaluate/eval_judge.py --model_type think (jbb-eval env)
python train/generative/EasyR1/scripts/model_merger.py --local_dir /path/to/ckpts/sft_cot_grpo/global_step_1800/actor
```

GRPO hyper-parameters live in `train/generative/grpo/run_grpo.sh` (rollout batch 32, n = 4, global batch 8, 1 epoch,  
checkpoint every 100 steps; knobs `DATA_DIR`, `REWARD_FN`, `N_GPU`, `EXTRA_ARGS`). Both stages resume from their latest  
checkpoint automatically.

## Benchmark construction

`construct/` holds the two data pipelines of the paper:

- **JudgeBiasBench**: `filter_helpsteer3.py` samples length-controlled pairs from HelpSteer3-Preference,
`process_bias.py` injects each bias (Gemini-2.0-Flash rewriting / identity prefixes / context and order manipulation),
followed by consistency filtering with Gemini-2.5-Pro. The released `data/eval/` files are the filtered sets.
- **Bias-augmented training negatives**: `bash construct/run_construction.sh` runs `gptout.py`
(GPT-4o bias-aware generation of extra rejected responses) and `verify_correctness.py` (GPT-4o verification).



## Citation

```bibtex
@article{ZHOU2027104801,
title = {Toward robust LLM-based judges: Taxonomic bias evaluation and debiasing optimization},
journal = {Information Fusion},
volume = {139},
pages = {104801},
year = {2027},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2026.104801},
url = {https://www.sciencedirect.com/science/article/pii/S1566253526006779},
author = {Hongli Zhou and Hui Huang and Rui Zhang and Kehai Chen and Bing Xu and Conghui Zhu and Tiejun Zhao and Muyun Yang}
}
```

## Acknowledgements

This repository benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1). Thanks for their wonderful works!