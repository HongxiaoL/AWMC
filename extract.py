import os
import pandas as pd
import yaml
from datetime import datetime
import builtins

# Dynamically get the current script directory to avoid hardcoding absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUMMARY_PATH = os.path.join(BASE_DIR, 'WBM', 'datasets', 'wbm-summary.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'WBM', 'models')
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')
TOTAL_LOG_PATH = os.path.join(BASE_DIR, 'log', 'total.log')

# --- Setup custom logging to total.log ---
original_print = builtins.print

def custom_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    original_print(msg, **kwargs)
    
    os.makedirs(os.path.dirname(TOTAL_LOG_PATH), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(TOTAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass

builtins.print = custom_print

def get_config():
    """Read full configuration from config.yaml"""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to read config.yaml: {e}")
        return {}

def get_top_percent_from_config(config=None):
    """Prioritize reading failure_top_percent from config.yaml"""
    if config is None:
        config = get_config()
    return config.get('failure_top_percent', None)

def get_random_seed_from_config(config=None):
    if config is None:
        config = get_config()
    return config.get('random_seed', 42)

def get_extract_rate_from_config(config=None):
    if config is None:
        config = get_config()
    return config.get('extract_rate', 0.1)

def extract(model, target_type='formation_energy', summary_path=DEFAULT_SUMMARY_PATH, top_percent=None):
    """
    Define and extract prediction failure/success samples (build error table)

    Parameters:
    - model (str): Model name, supports 10 models under WBM/models (e.g., 'pet', 'equflash', etc.).
    - target_type (str): Prediction target, 'formation_energy' or 'e_above_hull'.
    - summary_path (str): Path to the true label data wbm-summary.csv.
    - top_percent (float): Percentage of highest absolute errors to filter as failure. Prioritize reading from config.yaml, otherwise defaults to 0.05.

    Returns:
    - tuple(pd.DataFrame, pd.DataFrame): (failure_samples, success_samples)
    """
    # 0. Check if the output files already exist, if so skip extraction
    fail_dir = os.path.join(BASE_DIR, 'result', 'fail')
    succ_dir = os.path.join(BASE_DIR, 'result', 'succ')
    
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(succ_dir, exist_ok=True)
        
    fail_output_file = os.path.join(fail_dir, f"{model}_extract.csv")
    succ_output_file = os.path.join(succ_dir, f"{model}_extract.csv")
    
    if os.path.exists(fail_output_file) and os.path.exists(succ_output_file):
        print(f"[{model}] Output files already exist. Skipping extraction.")
        return None, None

    config = get_config()

    # Prioritize reading top_percent from config
    config_top_percent = get_top_percent_from_config(config=config)
    if config_top_percent is not None:
        top_percent = config_top_percent
    elif top_percent is None:
        top_percent = 0.05

    # Target definition dictionary
    target_cols_map = {
        'formation_energy': 'e_form_per_atom_mp2020_corrected',
        'e_above_hull': 'e_above_hull_mp2020_corrected_ppd_mp'
    }

    if target_type not in target_cols_map:
        raise ValueError(f"target_type must be one of {list(target_cols_map.keys())}")

    true_col = target_cols_map[target_type]

    # 1. Get regression targets
    print(f"[{model}] Step 1/6: Start reading true label file wbm-summary.csv ...")
    try:
        summary_df = pd.read_csv(summary_path, usecols=['material_id', true_col])
        print(f"[{model}] Step 1/6 Complete: Successfully read {len(summary_df)} true labels.")
    except FileNotFoundError:
        raise FileNotFoundError(f"True label file not found: {summary_path}. Please check the path.")

    # 2. Read model prediction data
    print(f"[{model}] Step 2/6: Start reading model prediction data ...")
    model_file = os.path.join(MODELS_DIR, f"{model}.csv")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model prediction file not found: {model_file}")

    try:
        predictions_df = pd.read_csv(model_file)
        if len(predictions_df.columns) == 1 and predictions_df.columns[0].strip().startswith('{'):
            predictions_df = pd.read_json(model_file)
    except Exception:
        predictions_df = pd.read_json(model_file)

    if 'material_id' not in predictions_df.columns:
        raise ValueError(f"{model}.csv must contain a 'material_id' column for merging. Available columns: {list(predictions_df.columns)}")
        
    print(f"[{model}] Step 2/6 Complete: Successfully read {len(predictions_df)} prediction records.")

    # Dynamically determine model prediction column name
    model_pred_col = None
    expected_col = f"e_form_per_atom_{model}"
    if expected_col in predictions_df.columns:
        model_pred_col = expected_col
    else:
        candidates = [c for c in predictions_df.columns if 'e_form' in c and 'uncorrected' not in c]
        if candidates:
            model_pred_col = candidates[0]
        else:
            raise ValueError(f"Cannot automatically find prediction column in {model}.csv (expected similar to '{expected_col}'). Currently available columns: {list(predictions_df.columns)}")

    # 3. Merge data
    print(f"[{model}] Step 3/6: Start merging true labels and prediction data ...")
    merged_df = pd.merge(predictions_df, summary_df, on='material_id', how='inner')

    if len(merged_df) == 0:
        raise ValueError(f"Merged data is empty. Please check the alignment rules for 'material_id' or the data content of {model}.")
        
    print(f"[{model}] Step 3/6 Complete: Obtained {len(merged_df)} valid records after merging.")

    random_seed = get_random_seed_from_config(config=config)
    sample_frac = get_extract_rate_from_config(config=config)
    print(f"[{model}] Step 4/7: Randomly sampling dataset (frac={sample_frac}) with random_seed={random_seed} ...")
    merged_df = merged_df.sample(frac=sample_frac, random_state=random_seed).reset_index(drop=True)
    print(f"[{model}] Step 4/7 Complete: Sampled {len(merged_df)} records.")

    # 5. Calculate error
    print(f"[{model}] Step 5/7: Calculating absolute error (abs_error) and squared error (sq_error) ...")
    merged_df['abs_error'] = (merged_df[model_pred_col] - merged_df[true_col]).abs()
    merged_df['sq_error'] = (merged_df[model_pred_col] - merged_df[true_col]) ** 2
    print(f"[{model}] Step 5/7 Complete: Error calculation finished.")

    # 6. Filter high-error (failure) and success samples
    print(f"[{model}] Step 6/7: Splitting success and failure samples based on config (top_percent={top_percent}) ...")
    total_samples = len(merged_df)
    top_n = max(1, int(total_samples * top_percent))

    # Sort by absolute error descending
    merged_df.sort_values(by='abs_error', ascending=False, inplace=True)
    
    high_error_samples = merged_df.iloc[:top_n].copy()
    success_samples = merged_df.iloc[top_n:].copy()
    
    print(f"[{model}] Step 6/7 Complete: Extracted {top_n} failure samples and {len(success_samples)} success samples.")

    # 7. Save results
    print(f"[{model}] Step 7/7: Saving results to result/fail and result/succ directories ...")
    high_error_samples.to_csv(fail_output_file, index=False)
    success_samples.to_csv(succ_output_file, index=False)

    threshold = high_error_samples['abs_error'].min()
    
    log_entry = {
        'type': 'EXTRACT',
        'model': model,
        'prediction_column': model_pred_col,
        'true_target_column': true_col,
        'total_samples': total_samples,
        'sample_frac': sample_frac,
        'random_seed': random_seed,
        'extracted_failure_ratio': f"{top_percent * 100:.2f}%",
        'failure_samples_count': top_n,
        'success_samples_count': len(success_samples),
        'absolute_error_threshold': f">= {threshold:.6f} eV/atom",
        'fail_output_file': fail_output_file,
        'succ_output_file': succ_output_file,
        'head_10_failure_samples': high_error_samples.head(10).to_dict(orient='records')
    }

    print(f"\n=== Samples Extraction Report ({model}) ===")
    print(yaml.dump([log_entry], allow_unicode=True, sort_keys=False, default_flow_style=False).strip())
    print("======================================================\n")

    return high_error_samples, success_samples

if __name__ == "__main__":
    config = get_config()
    models_to_run = config.get('models', [])
    
    if not models_to_run:
        print("No models configuration found in config.yaml, skipping extraction.")
    else:
        for model_name in models_to_run:
            print(f"\nProcessing model: {model_name}")
            try:
                extract(model_name)
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
