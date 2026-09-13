# audio-ml-lab

Modern ML architectures implemented from first principles in PyTorch and tested on audio.
The question behind every experiment here is *what actually happens when this idea meets sound?*

Core components (attention, LoRA, diffusion schedules, flow-matching objectives, contrastive losses)
are hand-written from the papers. Everything around them is
tooling.

## Experiments

| Experiment | Paper | Question | Status |
|---|---|---|---|
| `attention` | Vaswani et al. 2017 | Can a from-scratch Transformer encoder classify environmental audio? | PLANNED |
| `lora_rank` | Hu et al. 2021 | How low can LoRA rank go before the audio Transformer stops learning? | PLANNED |
| `diffusion_vs_flow` | Ho et al. 2020 · Lipman et al. 2022 | Same data, same budget — which trains better on spectrograms? | PLANNED |
| `contrastive` | Radford et al. 2021 | Can a small audio–text model do zero-shot audio classification? | PLANNED |

Status vocabulary: PLANNED → IN PROGRESS → IMPLEMENTED → TRAINED → BENCHMARKED → PUBLISHED.
Write-ups live at [novacodes.dev/papers](https://novacodes.dev/papers).

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/test_env.py          # environment + MPS/CUDA check
python scripts/get_esc50.py       # first dataset
pytest tests/test_attention.py    # fails until models/transformer/attention.py is written — that's the point
```
