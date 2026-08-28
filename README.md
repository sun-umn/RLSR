# Aligning Language Models with Selective Prediction

[![uv](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastral-sh%2Fuv%2Fmain%2Fassets%2Fbadge%2Fv0.json)](https://github.com/astral-sh/uv)
[![TRL](https://img.shields.io/badge/%F0%9F%A4%97%20TRL-0.23.1-yellow)](https://github.com/huggingface/trl)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO--2-1082C3)](https://www.deepspeed.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **[Aligning Language Models with Selective Prediction](https://openreview.net/forum?id=bqTLnXswlC)**  
> [Gaoxiang Luo](https://gaoxiangluo.github.io/), Yifan Wu, [Sinian Zhang](https://h7nian.github.io/), [Aryan Deshwal](https://aryandeshwal.github.io/), [Ju Sun](https://sunju.org)  
> _Second Workshop on Agents in the Wild: Safety, Security, and Beyond (ICML 2026)_

[[Paper](https://openreview.net/forum?id=bqTLnXswlC)] [[BibTeX](#citation)]

## Environment

Built on [RLCR](https://github.com/damanimehul/RLCR), which is vendored as a submodule at `RLCR/` — clone recursively:

```bash
git clone --recurse-submodules https://github.com/sun-umn/RLSR.git
```

Dependencies are managed by `uv`, pinned in `uv.lock` (python 3.12, torch 2.8.0, transformers 4.57.0, vllm 0.11.0, trl 0.23.1).

```bash
uv sync
# flash-attn must be built against the installed torch version:
MAX_JOBS=64 uv pip install flash-attn --no-build-isolation --force-reinstall --no-deps --no-cache --no-binary flash-attn
```

## Training (HotPotQA)

One-time: apply `rlcr.patch` to the submodule (trl 0.23 compatibility + answer-normalization fixes; see the patch header comments):

```bash
cd RLCR && git apply ../rlcr.patch && cd ..
```

Train (4× A100-40GB GPUs; select them via `CUDA_VISIBLE_DEVICES`):

```bash
export ACCELERATE_CONFIG_FILE=$PWD/deepspeed.yaml
uv run accelerate launch --num_processes 4 --config_file deepspeed.yaml \
  rl_runner.py --config configs/Qwen-7B/hotpot/RLSR.yaml   # or RLVR.yaml / RLCR.yaml
```

Equivalently, `./launch.sh <rlvr|rlcr|rlsr|rlvr-ta>` runs one method; outputs default to `data/<METHOD>-hotpot`. Runs log to the `RLSR` wandb project under your default entity — `wandb login` once first, or set `WANDB_MODE=offline`.

## Evaluation (HotPotQA)

`eval.sh` evaluates one checkpoint per invocation, following RLCR's official protocol and adding AURC / selective-prediction metrics. Run it on 2× A100-40GB GPUs (the generator and the LLM judge are each served tensor-parallel across them):

```bash
# <model> <dataset|all|all-extended> <gpu_ids> <method>
uv run bash eval.sh /path/to/checkpoint-300 all 0,1 rlsr
uv run bash eval.sh Qwen/Qwen2.5-7B all 0,1 base
```

- `method` must match how the model was trained: `rlvr` | `rlcr` | `rlsr` | `rlvr-ta` | `base`.
- `all` runs RLCR's 8 tasks (`hotpot-eval-em`, `hotpot-vanilla-eval-em`, `gsm8k`, `math-500`, `commonsenseqa`, `gpqa`, `simpleqa`, `trivia`); `all-extended` adds `medqa-eval` and `mbpp`, which need `uv run python data_scripts/medqa.py` and `data_scripts/mbpp.py` once first. Pass a single task name to run just one. (`mbpp` executes model-generated Python — run it only where that is safe.)
- Results land in `eval/results/<dataset>/` under the evaluated model; the full path is printed when the run finishes.

## Citation

```bibtex
@inproceedings{luo2026aligning,
  title = {Aligning Language Models with Selective Prediction},
  author = {Luo, Gaoxiang and Wu, Yifan and Zhang, Sinian and Deshwal, Aryan and Sun, Ju},
  booktitle = {Forty-third International Conference on Machine Learning Second Workshop on Agents in the Wild: Safety, Security, and Beyond},
  year = {2026},
  url = {https://openreview.net/forum?id=bqTLnXswlC},
}
```

## Acknowledgements

This codebase builds directly on [RLCR](https://github.com/damanimehul/RLCR), whose repo we vendor as a submodule and whose HotPotQA protocol we follow. Our rank-1 LoRA recipe follows Thinking Machines Lab's [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/).
