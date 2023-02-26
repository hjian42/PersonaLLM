# PersonaLLM
Explore whether LLMs are capable of simulating Big Five personalities

## generate personas

We first create male / female personas for each of 32 personality types.

```bash
for runname in run1 run2 run3 run4 run5
do
    python run_gender_bfi.py --output_folder ./outputs/gender_bfi/temp0.7/${runname}
done
```

## generate stories with personas

```bash
python run_gender_creative_writing.py
```