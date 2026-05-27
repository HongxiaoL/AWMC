import os
import glob
import pandas as pd
import numpy as np
import json
import concurrent.futures
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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


def process_model_feature(model, feature, summary_df, model_df, pred_col):
    fail_file = os.path.join(BASE_DIR, 'result', 'chara_fail', f'{model}_{feature}.csv')
    succ_file = os.path.join(BASE_DIR, 'result', 'chara_succ', f'{model}_{feature}.csv')
    
    if not os.path.exists(fail_file) or not os.path.exists(succ_file):
        return
        
    output_file = os.path.join(BASE_DIR, 'result', 'pair', f'paired_difference_{model}_{feature}.csv')
    if os.path.exists(output_file):
        print(f"[{model} | {feature}] Skipping: {output_file} already exists.")
        return
        
    print(f"\n--- Processing Model: {model}, Descriptor: {feature} ---")
    
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
        
    df = df.dropna(subset=[d_col, 'abs_error'] + X_cols).reset_index(drop=True)
    
    # Auto-encode categorical/string/boolean descriptors to numeric values
    if df[d_col].dtype == 'object' or df[d_col].dtype.name == 'category' or df[d_col].dtype == 'bool':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[d_col] = le.fit_transform(df[d_col].astype(str))
        print(f"[{model} | {feature}] Encoded categorical descriptor: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    if len(df) < 2:
        print(f"[{model} | {feature}] Not enough samples.")
        return
        
    X_val = df[X_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_val)
    
    n_neighbors = min(50, len(df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    
    d_val = df[d_col].values
    y_val = df['abs_error'].values
    
    pairs = []
    paired_j = set()
    
    for i in range(len(df)):
        if i in paired_j:
            continue
            
        best_j = -1
        best_dist = float('inf')
        
        for k in range(1, len(indices[i])):
            j = indices[i][k]
            dist = distances[i][k]
            
            if d_val[i] != d_val[j] and j not in paired_j:
                best_j = j
                best_dist = dist
                break
                
        if best_j != -1:
            if d_val[i] > d_val[best_j]:
                hi, lo = i, best_j
            else:
                hi, lo = best_j, i
                
            pairs.append({
                'hi_idx': hi,
                'lo_idx': lo,
                'hi_mat': df.loc[hi, 'material_id'],
                'lo_mat': df.loc[lo, 'material_id'],
                'hi_group': df.loc[hi, 'group'],
                'lo_group': df.loc[lo, 'group'],
                'hi_d': d_val[hi],
                'lo_d': d_val[lo],
                'hi_error': y_val[hi],
                'lo_error': y_val[lo],
                'delta_error': y_val[hi] - y_val[lo],
                'x_dist': best_dist
            })
            paired_j.add(hi)
            paired_j.add(lo)
            
    pairs_df = pd.DataFrame(pairs)
    print(f"[{model} | {feature}] Found {len(pairs_df)} valid pairs.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if len(pairs_df) == 0:
        pairs_df = pd.DataFrame(columns=['hi_idx', 'lo_idx', 'hi_mat', 'lo_mat', 'hi_group', 'lo_group', 'hi_d', 'lo_d', 'hi_error', 'lo_error', 'delta_error', 'x_dist'])
    pairs_df.to_csv(output_file, index=False)
    
    if len(pairs_df) > 0:
        mean_effect = pairs_df['delta_error'].mean()
        median_effect = pairs_df['delta_error'].median()
        print(f"[{model} | {feature}] Mean Δabs_error (high_d - low_d): {mean_effect:.6f}")
        print(f"[{model} | {feature}] Median Δabs_error: {median_effect:.6f}")
        
        import json
        effect_data = {
            "descriptor": feature,
            "model": model,
            "method": "Paired-difference",
            "n_pairs": int(len(pairs_df)),
            "mean_delta_abs_error": float(mean_effect),
            "median_delta_abs_error": float(median_effect),
            "unit": "eV/atom"
        }
        effect_json_path = os.path.join(BASE_DIR, 'result', 'report', f'pair_{model}_{feature}.json')
        os.makedirs(os.path.dirname(effect_json_path), exist_ok=True)
        with open(effect_json_path, 'w', encoding='utf-8') as f:
            json.dump(effect_data, f, indent=4)
        print(f"[{model} | {feature}] Pair JSON saved to {effect_json_path}")

def local_geometric_distortion_pair(chara_num=1, models=None, max_workers=2):
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
            
    print(f"Starting parallel paired-difference test for {len(tasks)} tasks...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_model_feature, m, f, s, md, p) for m, f, s, md, p in tasks]
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
        
    local_geometric_distortion_pair(chara_num=chara_num, models=models, max_workers=max_workers)
