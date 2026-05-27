# AWMC Repository Layout

Notes:
- `category_gen/`, `log/`, and `result/` are runtime-generated outputs. They may be absent in a fresh checkout.

| Entry | Description |
| --- | --- |
| category_agent/ | Researcher-agent feature extraction (writes `result/chara_*` and may generate scripts into `category_gen/`) |
| category_DML/ | Double Machine Learning (DML) causal estimation |
| category_pair/ | Paired-difference / matching-based causal estimation |
| category_report/ | LLM-written per-category reports (writes `result/report/`) |
| category_gen/ | Generated scripts (runtime output) |
| config/ | Basic parameters |
| data/feature.json | Default feature categories |
| raw/ | Original AI prediction results (CSV) |
| log/ | Logs (runtime output) |
| result/ | Pipeline outputs (runtime output) |
| result/fail/ | AI-failure samples |
| result/succ/ | AI-success samples |
| result/chara_fail/ | Extracted descriptors for failure samples |
| result/chara_succ/ | Extracted descriptors for success samples |
| result/pair/ | Paired-difference outputs |
| result/DML/ | DML outputs |
| result/report/ | Agent-written reports (JSON/MD) |
| result/allocated.json | Default and coordinator-appended features |
| result/final_report.md | Final consolidated report |
| WBM/ | The WBM dataset |
| coord.py | Coordinator agent |
| extract.py | Global error extraction / failure-sample selection |
| final_report.py | Final report aggregation (LLM) |
| init.sh | Main pipeline script |
| instruction.md | Task specification |
| llm_call.py | Unified LLM call wrapper |
| environment.yml | Conda environment |
| requirements.txt | Pip requirements |
