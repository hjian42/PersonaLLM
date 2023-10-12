# PersonaLLM
Explore whether LLMs are capable of simulating Big Five personalities

## generate personas

We first create 10 personas for each of 32 personality types.

```bash
conda activate audiencenlp
python3.9 run_bfi.py --model "GPT-3.5-turbo-0613"
python3.9 run_bfi.py --model "GPT-4-0613"

python3.9 run_creative_writing.py --model "GPT-3.5-turbo-0613"
python3.9 run_creative_writing.py --model "GPT-4-0613"
```

## generate stories with personas

```bash
python run_creative_writing.py
```