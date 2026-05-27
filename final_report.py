import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_call import call

def main():
    base_dir = Path(__file__).resolve().parent
    report_dir = base_dir / "result" / "report"
    output_dir = base_dir / "result"
    output_path = output_dir / "final_report.md"
    total_log_path = base_dir / "log" / "total.log"
    
    if output_path.exists():
        if os.getenv("AWMC_FORCE_REGEN") == "1":
            msg = f"Final report exists, but AWMC_FORCE_REGEN=1 so regenerating: {output_path}"
        else:
            msg = f"Final report already exists, skipping: {output_path}"
        print(msg)
        total_log_path.parent.mkdir(parents=True, exist_ok=True)
        with total_log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if os.getenv("AWMC_FORCE_REGEN") != "1":
            return

    if not report_dir.exists():
        print(f"Report directory does not exist: {report_dir}")
        return
        
    file_contents = []
    for ext in ["*.json", "*.md"]:
        for file_path in sorted(report_dir.glob(ext), key=lambda p: p.name):
            try:
                content = file_path.read_text(encoding="utf-8")
                file_contents.append(f"--- File: {file_path.name} ---\n{content}\n")
            except Exception as e:
                print(f"Failed to read file {file_path}: {e}")
                
    if not file_contents:
        print("No .json or .md files found.")
        return
        
    all_context = "\n".join(file_contents)
    
    prompt = f"""You are a professional AI materials science expert. I provide a collection of causal-analysis reports about how different AI models (e.g., TACE, Equflash) fail on materials-property prediction errors. The inputs include analysis documents (.md) and numeric result files (.json).

Task:
Analyze all provided files and produce a final consolidated report in English. The final report must show, for each AI model, the Top 5 causal "reason features" that lead to model failure (i.e., significantly larger prediction error), together with their decision strength. Use the causal effect values in the files such as ATE (DML) and mean_delta_abs_error (paired-difference). Positive and larger values indicate stronger increases in error.

Output rules (STRICT):
- Output MUST be valid Markdown only. Do NOT output any extra text outside the template.
- Follow the exact section headers and ordering shown in the template below.
- For every listed feature, you MUST include all required subfields (Effect sign, Decision strength, Evidence, Mechanism, Success-rate implication, Why it controls success/failure).
- Decision strength MUST explicitly state the sign (+ / -) and unit (eV/atom). If a value is not available, write "N/A".
- Evidence MUST cite the data source by filename(s) from the provided files (e.g., DML_*.json, pair_*.json, *_report.md).
- If DML and paired-difference disagree in sign, mark Effect sign as "Heterogeneous" and explain the conditions (e.g., "pairwise sub-regime vs global average").

Requirements:
1. Organize the report by model (one section per model).
2. For each model:
   - List the Top 5 features that INCREASE error (largest positive decision strength first). If fewer than 5 exist, list all.
   - Then (if available) list the Top 3 features that DECREASE error (most negative effect first) as "success-boosting" factors.
3. For each feature, include:
   - Feature name
   - Decision strength (quantitative value; summarize using DML ATE and/or paired-difference mean/median when available)
   - A short mechanism explanation based on the corresponding .md report(s)
   - A clear "success-rate implication" sentence that follows the SIGN of the causal effect:
     - If the causal effect on |error| is positive: "Higher X -> lower success rate (lower probability that |error| is small), because ..."
     - If the causal effect on |error| is negative: "Higher X -> higher success rate (higher probability that |error| is small), because ..."
     - Do not claim universal truth; explicitly tie the implication to the estimated causal effect in these reports.
   - A short explanation of why this feature determines the model's prediction success rate (i.e., why it increases/decreases the probability of small error), using concepts like distribution shift/OOD, missing physics (e.g., long-range electrostatics), symmetry-related degeneracy, relaxation sensitivity, or hidden electronic degrees of freedom.
4. Provide a final comparison across models (common vs. distinct failure drivers), and include a brief concluding paragraph explaining how these causal drivers translate into practical "success vs. failure" regions in materials space.

Template (copy exactly; replace bracketed parts):

# Final Causal Failure Report (Materials ML)

## 0. Executive Summary
- Models analyzed: [list]
- Global takeaway: [2-4 bullets]

## 1. Model: [MODEL_NAME]

### 1.1 Top 5 Error-Increasing Features (Failure Drivers)

#### 1.1.1 Feature: `[FEATURE_NAME]`
- Effect sign: [Positive|Negative|Heterogeneous]
- Decision strength: [e.g., +0.0202 eV/atom per +1σ (DML ATE); +0.0503 eV/atom (paired mean Δ|error|)]
- Evidence: [filenames]
- Mechanism: [2-5 sentences]
- Success-rate implication: [one sentence; must follow sign or "heterogeneous" conditions]
- Why it controls success/failure: [2-5 sentences]

(repeat Feature blocks for 1.1.2 ... 1.1.5)

### 1.2 Top 3 Error-Decreasing Features (Success-Boosting Factors)

#### 1.2.1 Feature: `[FEATURE_NAME]`
- Effect sign: [Positive|Negative|Heterogeneous]
- Decision strength: [...]
- Evidence: [...]
- Mechanism: [...]
- Success-rate implication: [...]
- Why it controls success/failure: [...]

(repeat Feature blocks for 1.2.2 ... 1.2.3; if none, write "N/A")

## 2. Cross-Model Comparison
- Common failure drivers: [bullets]
- Model-specific drivers: [bullets]
- Practical reliability rules of thumb: [bullets]

## 3. Practical Guidance (How to Use These Drivers)
- Risk-map features: [bullets]
- When to distrust predictions: [bullets]
- Suggested escalations: [bullets]

Here are the contents of all relevant files:
<files_content>
{all_context}
</files_content>
"""

    print("Calling LLM to generate the final report...")
    try:
        provider = os.getenv("AWMC_LLM_PROVIDER", "openai")
        model = os.getenv("AWMC_LLM_MODEL")
        if model:
            final_report_text = call(provider, model=model, input=prompt)
        else:
            final_report_text = call(provider, input=prompt)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_report_text, encoding="utf-8")
        saved_msg = f"Final report saved to: {output_path}"
        print(saved_msg)
        total_log_path.parent.mkdir(parents=True, exist_ok=True)
        with total_log_path.open("a", encoding="utf-8") as f:
            f.write("Calling LLM to generate the final report...\n")
            f.write(saved_msg + "\n")
        
    except Exception as e:
        print(f"LLM call failed: {e}")

if __name__ == "__main__":
    main()
