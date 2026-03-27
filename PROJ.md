# PatternFinder

## Python Environment

- Always use the project virtual environment: `.venv`
- Python: `.venv/bin/python`
- Pip: `.venv/bin/pip`

## Common Commands

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s dev/tests -v
PYTHONPATH=src .venv/bin/python -m pattern_finder --num-samples 1 --anomaly-type freq --series-length 128
PYTHONPATH=src .venv/bin/python dev/examples/generate_doc_example.py
```
