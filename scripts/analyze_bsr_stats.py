# -*- coding: utf-8 -*-
"""
从 results/ 的逐条评测结果做离线统计，不需重新推理：
  - BSR 的 Wilson 95% 置信区间
  - McNemar 配对显著性检验
  - 双向翻转数，正->错 / 错->正
  - 判别式模型的 chosen - rejected 分数 margin 变化

计数规则与论文一致：按字符串比较 original/bias_evaluation_correct；
BSR = #(True->False) / #(orig True)，Error 记录计入分母但不计为翻转；
生成式模型汇聚 12 类偏见，判别式模型汇聚 9 类；同名结果优先取简单命名的文件。
用法：python analyze_bsr_stats.py [--latex] [--verify]
"""
import argparse
import glob
import json
import math
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "baselines")

BIASES_9 = ["length", "authority", "beauty", "assertiveness", "sycophancy",
            "sentiment", "concreteness", "gender", "race"]
BIASES_12 = BIASES_9 + ["bandwagon", "superficial-reflection", "position"]

RM_MODELS = {"skywork-reward-v2-qwen3-8b", "skywork-reward-v2-llama-3.1-8b",
             "skywork-reward-llama-3.1-8b-v0.2", "llama-3.1-8b-base-rl-rm-rb2",
             "llama-3.1-8b-instruct-rm-rb2", "llama-3.1-tulu-3-8b-rl-rm-rb2",
             "grm-llama3-8b-instruct-rewardmodel-ft"}

# 论文中的逐 bias BSR，供 --verify 自检
PAPER_BSR = {
    "gpt-3.5-turbo": [66.0, 28.2, 41.1, 35.3, 39.7, 22.3, 45.5, 16.3, 21.7, 43.6, 13.5, 52.8],
    "claude-3-7-sonnet-20250219": [11.2, 3.1, 22.1, 5.9, 8.1, 5.5, 1.7, 10.3, 12.7, 19.5, 11.9, 9.6],
    "deepseek-r1": [22.5, 10.6, 17.6, 11.6, 9.2, 6.7, 10.0, 8.9, 13.7, 15.1, 11.4, 7.9],
    "o4-mini-2025-04-16": [30.5, 5.6, 13.7, 10.3, 14.8, 8.3, 5.6, 18.9, 25.5, 8.8, 8.5, 9.1],
    "kimi-k2-instruct": [13.0, 5.8, 12.2, 9.0, 11.9, 3.0, 1.4, 13.4, 16.9, 22.6, 8.7, 25.4],
    "qwen3-8b-no-thinking": [41.2, 12.9, 22.4, 17.7, 26.9, 10.6, 12.5, 21.7, 29.5, 21.7, 7.1, 45.8],
    "judgelm-7b": [59.1, 28.9, 35.9, 26.3, 38.8, 23.9, 30.7, 27.3, 40.4, 40.8, 9.7, 31.5],
    "auto-j-13b": [73.8, 42.4, 61.9, 45.5, 50.3, 32.7, 75.1, 18.1, 23.8, 31.6, 19.2, 15.5],
    "prometheus-7b-v2.0": [51.2, 62.5, 53.2, 68.4, 45.4, 51.4, 68.6, 27.8, 30.1, 25.2, 15.2, 30.8],
    "selene-1-mini-llama-3.1-8b": [23.9, 21.6, 47.9, 43.8, 30.0, 20.6, 25.9, 17.4, 19.5, 18.6, 15.6, 15.1],
    "skywork-reward-v2-qwen3-8b": [12.5, 3.5, 22.2, 7.0, 10.5, 6.5, 7.8, 19.0, 18.0],
    "skywork-reward-v2-llama-3.1-8b": [15.5, 3.6, 20.4, 6.3, 13.7, 4.7, 7.9, 15.2, 21.1],
    "skywork-reward-llama-3.1-8b-v0.2": [22.2, 6.6, 46.8, 21.6, 13.6, 14.9, 18.3, 23.2, 24.0],
    "llama-3.1-8b-base-rl-rm-rb2": [31.8, 11.7, 38.4, 20.6, 18.8, 9.3, 38.2, 35.2, 43.1],
    "llama-3.1-8b-instruct-rm-rb2": [38.2, 6.3, 35.2, 20.7, 19.8, 8.6, 18.9, 32.5, 39.3],
    "llama-3.1-tulu-3-8b-rl-rm-rb2": [30.8, 9.8, 40.4, 18.6, 20.5, 12.4, 14.8, 33.5, 55.0],
    "grm-llama3-8b-instruct-rewardmodel-ft": [21.4, 12.4, 50.2, 28.1, 16.6, 12.4, 26.8, 22.4, 20.3],
}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def mcnemar(b, c):
    """连续性校正 McNemar；返回 (chi2, p)，p 用 1 自由度卡方生存函数。"""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p


def pick_file(model_dir, bias):
    """优先简单命名，排除 summary/orig/bias 数据文件与 recalculated/helpsteer3 变体。"""
    cands = []
    for fp in glob.glob(os.path.join(model_dir, bias + "_*.jsonl")):
        base = os.path.basename(fp)
        if "summary" in base or base.endswith("_orig.jsonl") or base.endswith("_bias.jsonl"):
            continue
        cands.append(fp)
    if not cands:
        return None
    cands.sort(key=lambda x: ("recalculated" in x, "helpsteer3" in x))
    return cands[0]


def bias_stats(fp):
    """返回 (n, nt, tf, ft, dm_o, dm_b)；计数规则与论文一致。"""
    n = nt = tf = ft = 0
    dm_o, dm_b = [], []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "original_evaluation_correct" not in d or "bias_evaluation_correct" not in d:
                continue
            o = str(d["original_evaluation_correct"])
            b = str(d["bias_evaluation_correct"])
            n += 1
            if o == "True":
                nt += 1
                if b == "False":
                    tf += 1
            elif o == "False" and b == "True":
                ft += 1
            so, sb = d.get("scores_original"), d.get("scores_biased")
            if so and sb and d.get("label") in ("response1", "response2"):
                c = 0 if d["label"] == "response1" else 1
                dm_o.append(so[c] - so[1 - c])
                dm_b.append(sb[c] - sb[1 - c])
    return n, nt, tf, ft, dm_o, dm_b


def analyze_model(model_dir, name):
    biases = BIASES_9 if name in RM_MODELS else BIASES_12
    N = NT = TF = FT = 0
    dm_o, dm_b = [], []
    per_bias = []
    for bt in biases:
        fp = pick_file(model_dir, bt)
        if fp is None:
            return None
        n, nt, tf, ft, mo, mb = bias_stats(fp)
        if nt == 0:
            return None
        N += n
        NT += nt
        TF += tf
        FT += ft
        dm_o += mo
        dm_b += mb
        per_bias.append((bt, 100 * tf / nt))
    lo, hi = wilson_ci(TF, NT)
    chi2, p = mcnemar(TF, FT)
    r = {"n": N, "nt": NT, "cw": TF, "wc": FT, "bsr": TF / NT,
         "ci": (lo, hi), "chi2": chi2, "p": p, "per_bias": per_bias}
    if dm_o:
        r["margin_o"] = sum(dm_o) / len(dm_o)
        r["margin_b"] = sum(dm_b) / len(dm_b)
    return r


def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="逐 bias 对照论文 tab:bsr_all 数值自检")
    ap.add_argument("--results_dir", default=RESULTS_DIR,
                    help="directory of per-model result folders (default: results/baselines)")
    args = ap.parse_args()
    RESULTS_DIR = args.results_dir

    rows = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        model_dir = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(model_dir):
            continue
        r = analyze_model(model_dir, name)
        if r:
            rows.append((name, r))

    if args.verify:
        for name, r in rows:
            paper = PAPER_BSR.get(name)
            if not paper:
                print(f"{name}: 论文参考值缺失，跳过")
                continue
            bad = [(bt, v, pv) for (bt, v), pv in zip(r["per_bias"], paper)
                   if abs(round(v, 1) - pv) > 0.11]
            status = "OK" if not bad else f"MISMATCH {bad}"
            print(f"{name:42s} {status}")
        return

    if args.latex:
        for name, r in rows:
            dm = (f"{r['margin_o']:.2f} $\\to$ {r['margin_b']:.2f}"
                  if "margin_o" in r else "--")
            p = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            print(f"{name} & {100*r['bsr']:.1f} "
                  f"& [{100*r['ci'][0]:.1f}, {100*r['ci'][1]:.1f}] "
                  f"& {r['cw']} & {r['wc']} & {p} & {dm} \\\\")
    else:
        hdr = (f"{'model':42s} {'n':>5s} {'BSR':>6s} "
               f"{'95% CI':>14s} {'c->w':>5s} {'w->c':>5s} {'p':>8s} {'margin':>16s}")
        print(hdr)
        for name, r in rows:
            ci = f"[{100*r['ci'][0]:.1f},{100*r['ci'][1]:.1f}]"
            dm = (f"{r['margin_o']:.2f}->{r['margin_b']:.2f}"
                  if "margin_o" in r else "--")
            p = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            print(f"{name:42s} {r['n']:5d} "
                  f"{100*r['bsr']:6.1f} {ci:>14s} {r['cw']:5d} {r['wc']:5d} "
                  f"{p:>8s} {dm:>16s}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
