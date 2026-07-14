# HDC-RNN Algorithm 1 Code Bundle

This source-only bundle contains the execution chain used by `lag_V3_3.py`:

- `lag_V3_3.py`: screening workflow and lag-wise weight generation.
- `dataset/Data_CL.py`: synthetic CL trajectory generation.
- `models/Model_HoGRC.py`: reservoir-computing evaluator.
- `models/reservoir_model_HoGRC.py`: structured reservoir weight construction.
- `algorithm1_granger_screening.py`: standalone callback-based implementation of the Algorithm 1 screening logic.

No datasets, trained readouts, generated weights, or machine-specific paths are included. All inline comments, docstrings, and user-facing messages are in English. When `lag_V3_3.py` runs, generated artifacts are written only to `runtime/` next to this README.

Install dependencies with `pip install -r requirements.txt`, then run:

```powershell
python lag_V3_3.py
```
