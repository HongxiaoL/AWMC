import os
import sys
import json
import subprocess
import re
import pandas as pd
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv
import builtins

# Base directory of the project, avoiding hardcoded absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from llm_call import call

# Load environment variables (e.g., MP_API_KEY) so child processes can inherit them
load_dotenv(os.path.join(BASE_DIR, '.env'))

# --- Setup custom logging to total.log ---
original_print = builtins.print

def custom_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    original_print(msg, **kwargs)
    
    progress_match = re.search(r"Processed\s+(\d+)\s*/\s*(\d+)", msg)
    if progress_match:
        cur = int(progress_match.group(1))
        total = int(progress_match.group(2))
        if cur % 100 != 0 and cur != total:
            return

    log_file = os.path.join(BASE_DIR, "log", "total.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass

builtins.print = custom_print

def run_and_log_subprocess(cmd, timeout=None, env=None):
    """Run a subprocess, capture its stdout/stderr line by line, print and log it."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    output_lines = []
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                clean_line = line.rstrip('\n')
                print(clean_line)
                output_lines.append(clean_line)
        process.stdout.close()
        returncode = process.wait(timeout=timeout)
        return returncode, "\n".join(output_lines)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise

def process_model_feature(model, feature, fail_csv_path, succ_csv_path):
    print(f"\n--- [START] Processing Model: {model} | Feature: {feature} ---")
    safe_feature = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(feature)).strip("_")
    output_path = os.path.join(BASE_DIR, "category_gen", f"bonding_charge__{safe_feature}.py")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    to_run = []
    for split, csv_path in [("fail", fail_csv_path), ("succ", succ_csv_path)]:
        if not csv_path or not os.path.exists(csv_path):
            continue
        out_csv_path = os.path.join(BASE_DIR, "result", f"chara_{split}", f"{model}_{feature}.csv")
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        if os.path.exists(out_csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    in_len = sum(1 for _ in f) - 1
                df_out = pd.read_csv(out_csv_path)
                if feature in df_out.columns and len(df_out) >= in_len * 0.95 and not df_out[feature].isnull().all():
                    print(f"[{model} | {feature} | {split}] Complete CSV already exists ({len(df_out)} rows with data). Skipping this split.")
                    continue
            except Exception:
                pass
        to_run.append((split, csv_path, out_csv_path))

    if not to_run:
        return

    gen_split = "fail" if (fail_csv_path and os.path.exists(fail_csv_path)) else "succ"
    gen_csv_path = fail_csv_path if gen_split == "fail" else succ_csv_path
    gen_input_csv_filename = os.path.basename(gen_csv_path)
    gen_out_csv_path = os.path.join(BASE_DIR, "result", f"chara_{gen_split}", f"{model}_{feature}.csv")
    os.makedirs(os.path.dirname(gen_out_csv_path), exist_ok=True)

    skip_test_phase = False
    script_is_full = False
    code = ""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                code = f.read()
            if "material_ids[:10]" in code or "max_to_process = 10" in code:
                skip_test_phase = True
            else:
                skip_test_phase = True
                script_is_full = True
        except Exception:
            code = ""
            skip_test_phase = False
            script_is_full = False

    # Read the header of the CSV to provide context
    with open(gen_csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        first_row = f.readline().strip()
        
    base_prompt = f"""
I have a CSV file located at os.path.join(BASE_DIR, "result", "{gen_split}", "{gen_input_csv_filename}").
The first two lines of the CSV are:
{header}
{first_row}

I need to extract the material feature '{feature}' (which belongs to the category "Bonding & Charge") for each material in this CSV. 
Please write a complete Python script to extract or compute this feature for each `material_id` and save the result to a new CSV file.

CRITICAL INSTRUCTIONS:
1. Prioritize using `mp-api` (or `MPRester` from `mp_api.client`) to fetch the feature or structure if it is available via the API. If the feature cannot be directly extracted, you are ALLOWED and ENCOURAGED to write code that approximately computes or derives this feature based on the available structure or other properties.
2. If the feature requires a local structure calculation or for WBM materials, load the local structures from `WBM/datasets/wbm-init-structs.json`.
3. The local JSON file can be loaded via `pandas.read_json()`. It contains columns: `material_id`, `formula_from_cse`, and `initial_structure`.
4. The `initial_structure` column contains pymatgen structure dictionaries. Convert them to pymatgen Structure objects using `pymatgen.core.Structure.from_dict()`.
5. Match the structures with the `material_id`s in the CSV, calculate the feature, and save the final output.
6. The generated code MUST print log messages indicating the progress periodically (e.g., every 2 materials, print "Processed 2/10...").
7. DO NOT use any hardcoded absolute paths. Use `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` to dynamically resolve the project root.
   - The script MUST support overriding I/O paths via environment variables:
     - `INPUT_CSV_PATH` (full path to the input CSV)
     - `OUTPUT_CSV_PATH` (full path to the output CSV)
   - If env vars are not set, default to:
     - Input CSV path: `os.path.join(BASE_DIR, "result", "{gen_split}", "{gen_input_csv_filename}")`
     - Output CSV path: `os.path.join(BASE_DIR, "result", "chara_{gen_split}", "{model}_{feature}.csv")`
   - Input JSON path: `os.path.join(BASE_DIR, "WBM", "datasets", "wbm-init-structs.json")`
8. For this testing phase, process ONLY the FIRST 10 materials to verify the logic works. You MUST slice the materials list (e.g. `material_ids = material_ids[:10]`) so it finishes quickly.
9. Make sure to implement efficient error handling (e.g., skip structure calculation if it fails) to prevent the entire script from crashing midway.
10. Suppress all warnings by adding `import warnings; warnings.filterwarnings("ignore")` at the beginning of the script. 
11. **IMPORTANT: DO NOT use `MPRester.materials.summary.search` or `mpr.summary.search` as they cause severe DeprecationWarnings and nested asyncio errors that crash the parallel threads. Instead, strictly rely on local structure calculations or safe, non-deprecated APIs. DO NOT query the MP API using `mpr.summary.search`**.
12. **CRITICAL: BREAKPOINT CONTINUATION**. The script MUST support resuming from interruptions. Check if `{out_csv_path}` exists. If it does, read the `material_id`s already processed and SKIP them. Open the CSV in append mode (`mode='a'`) so results are saved incrementally. Write the header ONLY if creating the file for the first time.
13. **NO DUMMY CODE**: If the feature is ill-defined, ambiguous, impossible to calculate, or simply DOES NOT EXIST in typical material science properties or available databases, DO NOT generate dummy calculations. Instead, reply ONLY with the exact word "ILL_DEFINED" and do not generate any Python code.

Please return ONLY the valid Python code block, without any markdown formatting or explanations.
"""
    
    if not skip_test_phase:
        max_retries = 3
        try:
            import yaml
            with open(os.path.join(BASE_DIR, "config", "config.yaml"), "r", encoding="utf-8") as f:
                max_retries = yaml.safe_load(f).get("max_retries", 3)
        except Exception:
            pass

        success = False
        error_context = ""
        code = ""
        
        for attempt in range(max_retries):
            print(f"[{model} | {feature}] Attempt {attempt + 1}/{max_retries}...")
            
            prompt = base_prompt
            if error_context:
                prompt += f"\n\nIn the previous attempt, the generated code resulted in the following error:\n{error_context}\nPlease fix the code to resolve this error."
                
            print(f"[{model} | {feature}] Calling LLM...")
            response = call("openai", input=prompt)
            print(f"[{model} | {feature}] LLM Response received.")
            
            if "ILL_DEFINED" in response:
                print(f"[{model} | {feature}] LLM indicated feature is ill-defined. Rejecting request.")
                success = False
                break
                
            # Strip markdown code blocks if any
            code = response.strip()
            if code.startswith("```"):
                lines = code.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                code = '\n'.join(lines)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"[{model} | {feature}] Code saved to {output_path}")
            
            # Run the generated code
            print(f"[{model} | {feature}] Running {output_path}...")
            # Add a timeout so it doesn't hang forever during the testing phase
            try:
                env = os.environ.copy()
                env["INPUT_CSV_PATH"] = gen_csv_path
                env["OUTPUT_CSV_PATH"] = gen_out_csv_path
                returncode, stdout_text = run_and_log_subprocess([sys.executable, output_path], timeout=120, env=env)
            except subprocess.TimeoutExpired as e:
                success = False
                error_context = f"The testing script timed out after 120 seconds. It might be hanging on API calls or infinite loops."
                print(f"[{model} | {feature}] {error_context}")
                continue
            
            if returncode == 0:
                # Check the output CSV to see if the feature was properly extracted
                try:
                    df_out = pd.read_csv(gen_out_csv_path)
                    if feature not in df_out.columns:
                        success = False
                        error_context = f"The generated script finished, but the output CSV is missing the '{feature}' column."
                        print(f"[{model} | {feature}] {error_context}")
                    elif df_out[feature].isnull().all():
                        success = False
                        error_context = f"The generated script finished, but ALL extracted values for '{feature}' are empty/null. Check the computation logic; it might be silently failing and returning None."
                        print(f"[{model} | {feature}] {error_context}")
                    else:
                        print(f"[{model} | {feature}] Code executed and validated successfully!")
                        success = True
                        break
                except Exception as e:
                    success = False
                    error_context = f"The generated script finished, but failed to read the output CSV '{gen_out_csv_path}': {e}"
                    print(f"[{model} | {feature}] {error_context}")
            else:
                print(f"[{model} | {feature}] Execution failed.")
                error_context = f"Return code: {returncode}\nOUTPUT:\n{stdout_text}"
                print(f"[{model} | {feature}] {error_context}")
                
        if not success:
            print(f"[{model} | {feature}] Feature cannot be extracted.")
            return

    if not script_is_full:
        print(f"\n[{model} | {feature}] Testing phase successful. Now modifying code to process the entire dataset...")
        full_dataset_prompt = f"""
I have a Python script that successfully extracts the feature '{feature}' for the first 10 materials.
Here is the script:
```python
{code}
```

Please modify this script so that it processes the ENTIRE dataset instead of just the first 10 materials.
Remove the list slice `material_ids = material_ids[:10]` or any similar limits like `max_to_process = 10`.
CRITICAL: Since this script may run in parallel with others, DO NOT use tqdm. Using tqdm in parallel will garble the terminal output. Instead, use a simple `print` statement periodically (e.g., `if idx % 1000 == 0: print(f"[{model} | {feature}] Processed {{idx}}/{{total}}...")`) to log progress.
Make sure to keep the warning suppression (`import warnings; warnings.filterwarnings("ignore")`) at the top of the file.
CRITICAL: The script MUST support breakpoint continuation. Read existing `material_id`s from the output CSV (if it exists) to skip already processed ones. Append new results to the CSV incrementally (`mode='a'`) so progress is saved after every material. Write the header only when creating the file.
Do NOT change the paths or any other logic, just remove the limitation on the number of materials processed.

Please return ONLY the valid Python code block, without any markdown formatting or explanations.
"""
        print(f"[{model} | {feature}] Calling LLM to generate full script...")
        full_response = call("openai", input=full_dataset_prompt)
        
        full_code = full_response.strip()
        if full_code.startswith("```"):
            lines = full_code.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            full_code = '\n'.join(lines)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
        print(f"[{model} | {feature}] Full processing code saved to {output_path}")
        print(f"[{model} | {feature}] Running {output_path} for full dataset...")
    
    for split, csv_path, out_csv_path in to_run:
        env = os.environ.copy()
        env["INPUT_CSV_PATH"] = csv_path
        env["OUTPUT_CSV_PATH"] = out_csv_path
        returncode, _ = run_and_log_subprocess([sys.executable, output_path], env=env)
        if returncode == 0:
            print(f"[{model} | {feature} | {split}] Dataset processed successfully!")
        else:
            print(f"[{model} | {feature} | {split}] Dataset execution failed with return code: {returncode}")

def bonding_charge_extract(chara_num=1, models=None, max_workers=2):
    if models is None:
        models = ["pet"]

    allocated_json_path = os.path.join(BASE_DIR, "result", "allocated.json")
    
    with open(allocated_json_path, 'r', encoding='utf-8') as f:
        allocated = json.load(f)
        
    category_features = allocated.get("Bonding & Charge", [])
    features_to_extract = category_features[:chara_num]
    
    tasks = []
    for model in models:
        fail_csv_path = os.path.join(BASE_DIR, "result", "fail", f"{model}_failure.csv")
        if not os.path.exists(fail_csv_path):
            alt_fail = os.path.join(BASE_DIR, "result", "fail", f"{model}_extract.csv")
            fail_csv_path = alt_fail if os.path.exists(alt_fail) else None

        succ_csv_path = os.path.join(BASE_DIR, "result", "succ", f"{model}_success.csv")
        if not os.path.exists(succ_csv_path):
            alt_succ = os.path.join(BASE_DIR, "result", "succ", f"{model}_extract.csv")
            succ_csv_path = alt_succ if os.path.exists(alt_succ) else None

        if not fail_csv_path and not succ_csv_path:
            print(f"Warning: No input CSV found for model {model} in result/fail or result/succ. Skipping model.")
            continue

        for feature in features_to_extract:
            tasks.append((model, feature, fail_csv_path, succ_csv_path))
            
    print(f"Starting parallel extraction for {len(tasks)} tasks with max_workers={max_workers}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_model_feature, m, f, cf, cs) for m, f, cf, cs in tasks]
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
        print(f"Failed to load config from {config_path}: {e}")
        chara_num = 1
        models = ["pet"]
        max_workers = 2
        
    bonding_charge_extract(chara_num=chara_num, models=models, max_workers=max_workers)
