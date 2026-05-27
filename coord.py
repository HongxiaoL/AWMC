import json
import os
import random
import yaml
from datetime import datetime
from llm_call import call

def coordinator():
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_file = os.path.join(base_dir, 'data', 'feature.json')
    result_dir = os.path.join(base_dir, 'result')
    result_file = os.path.join(result_dir, 'allocated.json')
    config_file = os.path.join(base_dir, 'config', 'config.yaml')
    log_file = os.path.join(base_dir, 'log', 'total.log')

    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            existing_allocated = json.load(f)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] [INFO] [COORDINATOR] allocated.json exists, skipping allocation."
        print(msg)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
        return existing_allocated

    # Read configuration file
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    random_seed = config['random_seed']
    allocate_sample_ratio = config['allocate_sample_ratio']
    new_chara_num = config['new_chara_num']

    # Set random seed
    random.seed(random_seed)

    # 1. Read features from feature.json
    with open(feature_file, 'r', encoding='utf-8') as f:
        features = json.load(f)

    # 2. Randomly select feature descriptors by ratio
    selected_features = {}
    for category, items in features.items():
        sample_size = max(1, int(len(items) * allocate_sample_ratio)) if items else 0
        selected_features[category] = random.sample(items, sample_size)

    # 3. Build prompt and call LLM to generate new features
    prompt = f"""
I have a set of material science or chemistry features categorized into several categories.
Here are the selected categories and features:
{json.dumps(selected_features, indent=2, ensure_ascii=False)}

Please generate exactly {new_chara_num} NEW and DIFFERENT features that are not in the list above.
CRITICAL CONSTRAINT: You are allowed to propose features that can be approximately calculated (e.g., via empirical formulas, machine learning surrogates, or low-cost approximations). However, you MUST STRICTLY AVOID any features that require expensive ab initio / from-scratch calculations (like DFT) or Molecular Dynamics (MD) simulations.
You MUST classify these {new_chara_num} new features into the EXACT SAME existing categories provided above. Do NOT create any new categories.
Return ONLY a valid JSON object, where keys are ONLY the existing category names and values are lists of string features. Do not output any markdown formatting like ```json or ```, just the raw JSON text.
"""

    # Call the call function from llm_call.py with default configuration
    print(f"Calling LLM to generate {new_chara_num} new features...")
    response = call(provider="openai", input=prompt)

    # 4. Parse JSON returned by LLM
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    try:
        new_features = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM response:")
        print(response)
        raise e

    # Log: count, content and classification of new features
    total_new_features = sum(len(items) for items in new_features.values())
    
    print("-" * 40)
    print(f"Successfully generated {total_new_features} new features:\n")
    print(json.dumps(new_features, indent=4, ensure_ascii=False))
    print("-" * 40)
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [INFO] [COORDINATOR] Generated {total_new_features} new features: {json.dumps(new_features, ensure_ascii=False)}"
        f.write(log_msg + "\n")

    # 5. Merge original sampled features and newly generated features
    final_features = {}
    # Copy selected features first
    for cat in features.keys():
        final_features[cat] = selected_features.get(cat, []).copy()

    # Add newly generated features
    for cat, items in new_features.items():
        if cat in final_features:
            for item in items:
                if item not in final_features[cat]:
                    final_features[cat].append(item)
        else:
            final_features[cat] = items

    # 6. Save to result/allocated.json
    os.makedirs(result_dir, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(final_features, f, indent=4, ensure_ascii=False)

    print(f"Successfully merged features and saved to {result_file}")
    return final_features

if __name__ == "__main__":
    coordinator()