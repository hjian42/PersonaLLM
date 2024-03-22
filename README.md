# PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2201.07281-b31b1b.svg)](https://arxiv.org/abs/2305.02547)

This repo contains the official code for running `LLM persona` experiments and subsequent analyses in the PersonaLLM paper.


## Simulate `LLM personas`

We first create 10 personas for each of 32 personality types.

```bash
conda activate audiencenlp
python3.9 run_bfi.py --model "GPT-3.5-turbo-0613"
python3.9 run_bfi.py --model "GPT-4-0613"
python3.9 run_bfi.py --model "llama-2"
```

## Generate stories with `LLM personas`

```bash
python3.9 run_creative_writing.py --model "GPT-3.5-turbo-0613"
python3.9 run_creative_writing.py --model "GPT-4-0613"
python3.9 run_creative_writing.py --model "llama-2"
```

## References

If you use this repository in your research, please kindly cite [our paper](https://arxiv.org/abs/2305.02547): 

```bibtex
@article{jiang2023personallm,
  title={PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits},
  author={Hang Jiang and Xiajie Zhang and Xubo Cao and Cynthia Breazeal and Jad Kabbara and Deb Roy},
  booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
  year={2024}
}
```

## Acknowledgement

PersonaLLM is a research program from MIT Center for Constructive Communication (@mit-ccc), MIT Media Lab, and Stanford University. We are interested in drawing from social and cognitive sciences to understand the behaviors of foundation models. 