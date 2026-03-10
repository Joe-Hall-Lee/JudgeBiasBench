# Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization

This is the official repository for paper [Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization](https://arxiv.org/abs/2603.08091).

In this paper, we introduce JudgeBiasBench, a benchmark designed to evaluate the biases in LLM-based judges.

---

## 📂 Repository Structure

```text
JudgeBiasBench/
├─ data/
│  └─ eval/                # JudgeBiasBench and general judge benchmarks
│     ├─ reward_bench/
│     │  ├─ evalbiasbench.jsonl
│     │  ├─ rewardbench.jsonl
│     │  ├─ judgebench.jsonl
│     │  ├─ rm-bench.jsonl
│     │  └─ rmb_pairwise.jsonl
│     ├─ length_bias.jsonl
│     ├─ authority_bias.jsonl
│     ├─ beauty_bias.jsonl
│     ├─ assertiveness_bias.jsonl
│     ├─ sycophancy_bias.jsonl
│     ├─ sentiment_bias.jsonl
│     ├─ concreteness_bias.jsonl
│     ├─ gender_bias.jsonl
│     ├─ race_bias.jsonl
│     ├─ bandwagon_bias.jsonl
│     ├─ superficial-reflection_bias.jsonl
│     ├─ position_bias.jsonl
│     └─ refinement-aware_bias.jsonl
├─ src/
│  ├─ construct/            # data construction utilities
│  ├─ evaluate/             # main evaluation entrypoints and handlers
│  │  ├─ eval_judge.py
│  │  ├─ generic_benchmark_handler.py
│  │  ├─ rewardbench_handler.py
│  │  ├─ rmbench_handler.py
│  │  ├─ chateval_handler.py
│  │  ├─ calibration_handler.py
│  │  ├─ model_utils.py
│  │  └─ build_prompt.py
│  └─ train/                # reward model training
├─ scripts/
│  └─ run_evaluation_judge.sh
├─ results/                 # per-model outputs and summaries
├─ requirements.txt
└─ README.md

```

## ⚡️ Usage

### Preparation

Please refer to the following commands to prepare your environment.

```shell
conda create -n rm-distiller python=3.12
pip install -r requirements.txt
```

### Evaluation

Evaluate a specific bias type:

```bash
python src/evaluate/eval_judge.py <bias_type> \
  --model_path </path/to/vllm-compatible-model> \
  --model_name <your_model_tag> \
  --model_type <strategy> \
  --temperature 0.0 \
  --max_new_token 2048 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.9
```

Evaluate all biases:

```bash
python src/evaluate/eval_judge.py all \
  --model_path </path/to/model> \
  --model_name <your_model_tag> \
  --model_type <strategy> \
  --temperature 0.0 \
  --max_new_token 2048 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.9
```

## Citation

```bibtex
@misc{zhou2026robustllmbasedjudgestaxonomic,
      title={Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization},
      author={Hongli Zhou and Hui Huang and Rui Zhang and Kehai Chen and Bing Xu and Conghui Zhu and Tiejun Zhao and Muyun Yang},
      year={2026},
      eprint={2603.08091},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.08091},
}
```
