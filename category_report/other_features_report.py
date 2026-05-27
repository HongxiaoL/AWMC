import os
import json
import pandas as pd
import concurrent.futures
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import builtins
from datetime import datetime

# --- Setup custom logging to total.log ---
original_print = builtins.print

def custom_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    original_print(msg, **kwargs)
    
    log_file = os.path.join(BASE_DIR, "log", "total.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass

builtins.print = custom_print

sys.path.append(BASE_DIR)
from llm_call import call

def process_report(model, feature):
    csv_path = os.path.join(BASE_DIR, 'result', 'pair', f'paired_difference_{model}_{feature}.csv')
    if not os.path.exists(csv_path):
        return
        
    report_path = os.path.join(BASE_DIR, 'result', 'report', f'{model}_{feature}_report.md')
    if os.path.exists(report_path):
        print(f"[{model} | {feature}] Skipping: {report_path} already exists.")
        return
        
    print(f"\n--- Generating report for Model: {model}, Descriptor: {feature} ---")
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"[{model} | {feature}] Empty pairs file, generating 'cannot match' report.")
        cannot_match_content = f"""# Causal Test Report for `{feature}` on model `{model}`

**Method**: Paired-difference causal test

**Result**: Cannot match valid pairs.

**Explanation**: 
No valid sample pairs could be found where the descriptor `{feature}` clearly differs while keeping all other confounders (like volume, number of sites, bandgap) sufficiently similar. 
This typically indicates one of two things:
1. The descriptor `{feature}` is highly collinear with the matched confounders, meaning any change in this descriptor inevitably leads to large changes in the confounders.
2. There is insufficient variation or sample overlap between the different values of this descriptor in the given dataset.

Due to the lack of valid pairs, the paired-difference causal effect cannot be estimated.
"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(cannot_match_content)
        return
        
    mean_eff = df['delta_error'].mean()
    median_eff = df['delta_error'].median()
    n_pairs = len(df)
    

        
    dml_json_path = os.path.join(BASE_DIR, 'result', 'report', f'DML_{model}_{feature}.json')
    dml_ate = None
    if os.path.exists(dml_json_path):
        with open(dml_json_path, 'r', encoding='utf-8') as f:
            dml_data = json.load(f)
        dml_ate = dml_data.get("ATE")

    prompt = f"""
You are an expert in materials science and machine learning.
We are analyzing the prediction errors of the AI model '{model}' on the WBM dataset.
We performed causal tests on the descriptor '{feature}' using two methods:
1. Paired-difference causal test: We matched materials with similar basic properties (volume, number of sites, bandgap) but different values for '{feature}'.
2. Double Machine Learning (DML): We used machine learning to partial out the effect of confounders.

We defined 'Treated' as having a higher value for '{feature}', and 'Control' as having a lower value.

The results from the Paired-difference test ({n_pairs} matched pairs) are:
- Mean Δabs_error (High value error - Low value error): {mean_eff:.6f} eV/atom
- Median Δabs_error: {median_eff:.6f} eV/atom
"""

    if dml_ate is not None:
        prompt += f"""
The results from the DML test are:
- Average Treatment Effect (ATE) per 1 standard deviation increase in '{feature}': {dml_ate:.6f} eV/atom
"""
    else:
        prompt += f"""
The results from the DML test are currently unavailable.
"""

    prompt += f"""
Please write a short conclusion report in Markdown format. Following these instructions:
- Cause: the physical/chemical meaning of descriptor '{feature}'.
- Effect: direction and magnitude of error increase/decrease consistently found across the available causal methods. (e.g., higher '{feature}' -> higher/lower error).
- Mechanistic hypothesis: why this feature implies stronger out-of-distribution (OOD) chemistry/structure/relaxation behavior, leading to model failure. 

Output ONLY the Markdown text. Use English for the report content.
"""
    
    print(f"[{model} | {feature}] Calling LLM to generate report...")
    try:
        report_md = call("openai", input=prompt)
    except Exception as e:
        print(f"[{model} | {feature}] OpenAI failed: {e}. Trying Google provider...")
        try:
            report_md = call("google", input=prompt)
        except Exception as e2:
            print(f"[{model} | {feature}] Google failed: {e2}. Skipping.")
            return
            
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    print(f"[{model} | {feature}] Report saved to: {report_path}")

def other_features_report(chara_num=1, models=None, max_workers=2):
    if models is None:
        models = ["pet"]

    allocated_json_path = os.path.join(BASE_DIR, "result", "allocated.json")
    with open(allocated_json_path, 'r', encoding='utf-8') as f:
        allocated = json.load(f)
        
    category_features = allocated.get("Other Features", [])
    features_to_process = category_features[:chara_num]
    
    tasks = []
    for model in models:
        for feature in features_to_process:
            tasks.append((model, feature))
            
    print(f"Starting parallel report generation for {len(tasks)} tasks...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_report, m, f) for m, f in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"A task failed with exception: {e}")

if __name__ == "__main__":
    import yaml
    config_path = os.path.join(BASE_DIR, "config", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        chara_num = config.get("chara_num", 1)
        models = config.get("models", ["pet"])
        max_workers = config.get("max_workers", 2)
    except Exception as e:
        chara_num = 1
        models = ["pet"]
        max_workers = 2
        
    other_features_report(chara_num=chara_num, models=models, max_workers=max_workers)
