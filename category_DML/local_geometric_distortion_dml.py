import os
import glob
import pandas as pd
import numpy as np
import json
import concurrent.futures
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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


def process_model_feature_dml(model, feature, summary_df, model_df, pred_col):
    fail_file = os.path.join(BASE_DIR, 'result', 'chara_fail', f'{model}_{feature}.csv')
    succ_file = os.path.join(BASE_DIR, 'result', 'chara_succ', f'{model}_{feature}.csv')
    
    if not os.path.exists(fail_file) or not os.path.exists(succ_file):
        return
        
    output_file = os.path.join(BASE_DIR, 'result', 'DML', f'DML_{model}_{feature}.csv')
    if os.path.exists(output_file):
        print(f"[{model} | {feature}] Skipping: {output_file} already exists.")
        return
        
    print(f"\n--- Processing Model: {model}, Descriptor: {feature} (DML) ---")
    
    fail_df = pd.read_csv(fail_file)
    succ_df = pd.read_csv(succ_file)
    
    fail_df['group'] = 'fail'
    succ_df['group'] = 'succ'
    
    pool_df = pd.concat([fail_df, succ_df], ignore_index=True)
    
    df = pool_df.merge(summary_df, on='material_id', how='inner')
    df = df.merge(model_df[['material_id', pred_col]], on='material_id', how='inner')
    
    df['abs_error'] = (df[pred_col] - df['e_form_per_atom_mp2020_corrected']).abs()
    
    d_col = feature
    X_cols = ['n_sites', 'volume', 'bandgap_pbe', 'site_stats_fingerprint_init_final_norm_diff']
    X_cols = [c for c in X_cols if c != d_col]
    
    if d_col not in df.columns:
        print(f"[{model} | {feature}] Skipping: descriptor '{d_col}' not found.")
        return
        
    # Drop rows with missing values
    df = df.dropna(subset=[d_col, 'abs_error'] + X_cols).reset_index(drop=True)
    
    # Auto-encode categorical/string/boolean descriptors to numeric values
    if df[d_col].dtype == 'object' or df[d_col].dtype.name == 'category' or df[d_col].dtype == 'bool':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[d_col] = le.fit_transform(df[d_col].astype(str))
        print(f"[{model} | {feature}] Encoded categorical descriptor: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    if len(df) < 50:
        print(f"[{model} | {feature}] Not enough samples for DML.")
        return
        
    X_val = df[X_cols].values
    T_val = df[d_col].values
    Y_val = df['abs_error'].values
    
    # Handle categorical descriptor intuitively or let RF handle it if numerical
    # We assume d_col is numeric since DML primarily targets continuous/binary numerical variables.
    
    # K-Fold Cross-fitting
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_tilde = np.zeros(len(df))
    t_tilde = np.zeros(len(df))
    
    model_y = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    model_t = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    
    for train_idx, test_idx in kf.split(X_val):
        X_train, X_test = X_val[train_idx], X_val[test_idx]
        Y_train, Y_test = Y_val[train_idx], Y_val[test_idx]
        T_train, T_test = T_val[train_idx], T_val[test_idx]
        
        model_y.fit(X_train, Y_train)
        model_t.fit(X_train, T_train)
        
        y_hat = model_y.predict(X_test)
        t_hat = model_t.predict(X_test)
        
        y_tilde[test_idx] = Y_test - y_hat
        t_tilde[test_idx] = T_test - t_hat
        
    # OLS for the residual
    ols = LinearRegression(fit_intercept=False)
    ols.fit(t_tilde.reshape(-1, 1), y_tilde)
    raw_ate = ols.coef_[0]
    
    # Scale ATE by 1 Standard Deviation of T to make it interpretable across different feature scales
    T_std = np.std(T_val)
    ate = raw_ate * T_std if T_std > 0 else raw_ate
    
    print(f"[{model} | {feature}] DML ATE (per 1 std): {ate:.6f} (Raw ATE: {raw_ate:.6f})")
    
    # Save the detailed sample-level residuals to CSV instead of just one line
    result_df = pd.DataFrame({
        'material_id': df['material_id'],
        'group': df['group'],
        'T_actual': T_val,
        'Y_actual': Y_val,
        'T_residual': t_tilde,
        'Y_residual': y_tilde
    })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"[{model} | {feature}] DML sample residuals saved to {output_file}")
    
    # Save the ATE result to report JSON
    effect_data = {
        "descriptor": feature,
        "model": model,
        "method": "Double Machine Learning",
        "n_samples": len(df),
        "ATE": float(ate),
        "unit": "eV/atom"
    }
    
    effect_json_path = os.path.join(BASE_DIR, 'result', 'report', f'DML_{model}_{feature}.json')
    os.makedirs(os.path.dirname(effect_json_path), exist_ok=True)
    with open(effect_json_path, 'w', encoding='utf-8') as f:
        json.dump(effect_data, f, indent=4)
    print(f"[{model} | {feature}] DML ATE JSON saved to {effect_json_path}")

def local_geometric_distortion_dml(chara_num=1, models=None, max_workers=2):
    if models is None:
        models = ["pet"]

    allocated_json_path = os.path.join(BASE_DIR, "result", "allocated.json")
    with open(allocated_json_path, 'r', encoding='utf-8') as f:
        allocated = json.load(f)
        
    category_features = allocated.get("Local Geometric Distortion", [])
    features_to_process = category_features[:chara_num]
    
    summary_df = pd.read_csv(os.path.join(BASE_DIR, 'WBM', 'datasets', 'wbm-summary.csv'))
    
    tasks = []
    for model in models:
        model_csv_path = os.path.join(BASE_DIR, 'WBM', 'models', f'{model}.csv')
        if not os.path.exists(model_csv_path):
            continue
        model_df = pd.read_csv(model_csv_path)
        pred_col = f'e_form_per_atom_{model}'
        if pred_col not in model_df.columns:
            pred_cols = [c for c in model_df.columns if 'e_form_per_atom' in c and 'uncorrected' not in c]
            if pred_cols:
                pred_col = pred_cols[0]
            else:
                continue
                
        for feature in features_to_process:
            tasks.append((model, feature, summary_df, model_df, pred_col))
            
    print(f"Starting parallel DML test for {len(tasks)} tasks...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_model_feature_dml, m, f, s, md, p) for m, f, s, md, p in tasks]
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
        
    local_geometric_distortion_dml(chara_num=chara_num, models=models, max_workers=max_workers)
