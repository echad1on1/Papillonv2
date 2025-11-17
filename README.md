# Papillon — Efficient and Stealthy Fuzz Testing–Powered Jailbreaks for LLMs

Official repository for the paper "PAPILLON: Efficient and Stealthy Fuzz Testing–Powered Jailbreaks for LLMs" (Usenix Security 2025).

The paper is available at: https://arxiv.org/abs/2409.14866

## ✨Overview

Papillon implements the fuzz-testing pipeline, datasets, and evaluation harness used in the paper. It provides scripts to reproduce experiments, run the two-stage judge, and evaluate attack success rates.

![overview.png](./overview.png)

## Quick Links

- Paper: https://arxiv.org/abs/2409.14866
- Code: this repository
- Datasets: `datasets/questions/`

## 💥Features

- Reproducible fuzz-testing pipeline for LLM jailbreak evaluation
- Two-stage automated judging (local RoBERTa judge + LLM judge)
- Predefined datasets and an interface to add custom questions

## ⚒️Requirements

- Python 3.10+
- PyTorch (tested with `2.1.2+cu12.1`)

Core Python packages used in the repository (examples):

```bash
# requirements
pip install "fschat[model_worker,webui]"
pip install vllm 
pip install openai                # for openai LLM
pip install termcolor
pip install openpyxl
pip install google-generativeai   # for google PALM-2
pip install anthropic  # for anthropic
```

## Installation

1. Clone the repository.
2. Install dependencies (see Requirements).

## 🤖Models

1. First-stage judge: we use a finetuned RoBERTa-large model [huggingface](https://huggingface.co/hubert233/GPTFuzz) from [GPTFuzz](https://github.com/sherdencooper/GPTFuzz) as our first-level judge model. Please download it to "./roberta".
2. Second-stage judge: an LLM client (OpenAI or other provider). Configure API credentials in `Judge/language_models.py` where the client is instantiated. Example:

```python
# line 106 in ./Judge/language_models.py
client = OpenAI(base_url="[your proxy url(if use)]", api_key="your api key", timeout = self.API_TIMEOUT)
```

## 🗂️Datasets

We have 3 available datasets to jailbreak:

1. `datasets/questions/question_target_list.csv` : sampled from two public datasets: [llm-jailbreak-study](https://sites.google.com/view/llm-jailbreak-study) and [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf). Following the format of [GCG](https://github.com/llm-attacks/llm-attacks), we have added corresponding target for each question.
2. `datasets/questions/question_target.csv` : advbench.
3. `datasets/questions/question_target_custom.csv` : subset of advbench.

## 🚀Quickstart

Run a simple experiment (example uses OpenAI model as the target):

```bash
python run.py --openai_key YOUR_OPENAI_KEY --model_path gpt-3.5-turbo --target_model gpt-3.5-turbo
```

Adjust flags as needed for your environment and target model.

## ⚙️Evaluation

To evaluate results, set `directory_path` to the directory containing experiment outputs and run:

```bash
python eval.py
```

This will compute metrics such as ASR and AQ used in the paper.

## Responsible Use

This repository contains research code intended for academic and defensive purposes (vulnerability analysis, robustness evaluation, and mitigation research). Do not use these tools to harm, exploit, or bypass safety systems in production models or services.

By using this code you agree to comply with applicable laws and the terms of service of any platform or API you use. If you perform experiments on third-party systems, obtain explicit authorization first.

## 📖Acknowledgements

This implementation builds on and reuses ideas and components from prior works, in particular [GPTFuzz](https://github.com/sherdencooper/GPTFuzz) and [PAIR/jailbreakingllms](https://github.com/patrickrchao/jailbreakingllms). Thanks to the authors of those excellent projects.

## 📌Citation

If you use this code or datasets in your research, please cite the paper:

```bibtex
@article{gong2024effective,
  title={Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs},
  author={Gong, Xueluan and Li, Mingzhe and Zhang, Yilin and Ran, Fengyuan and Chen, Chen and Chen, Yanjiao and Wang, Qian and Lam, Kwok-Yan},
  journal={arXiv preprint arXiv:2409.14866},
  year={2024}
}
```
