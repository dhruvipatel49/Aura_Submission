#!/usr/bin/env python3
"""
PRE-SUBMISSION RIGOROUS VERIFICATION SCRIPT:
1. Check saved model artifact (file size, timestamp, feature list, feature count).
2. Co-location guard cost analysis on true matches vs false positives.
3. Fresh computation of in-domain (US/India) validation Macro F_0.5 from scratch.
"""

import os, sys, time, gc, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from normalize import normalize_name, normalize_address
from features import compute_pair_features, compute_features_batch
from model import load_model, predict_scores, compute_f05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "models", "lgbm_model.pkl")

print("=" * 80)
print("1. MODEL CONSISTENCY & ARTIFACT METADATA CHECK")
print("=" * 80)

model_stat = os.stat(MODEL_PATH)
print(f"Model file path: {MODEL_PATH}")
print(f"Model file size: {model_stat.st_size} bytes")
print(f"Model file modification time: {time.ctime(model_stat.st_mtime)}")

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_cols = model_data['feature_cols']
base_thresh = model_data['threshold']

print(f"Number of features in model: {len(feature_cols)}")
print(f"Feature list: {feature_cols}")
print(f"Saved base threshold: {base_thresh}")

# Check whether any flagged length/count features exist in model
flagged_features = {
    'name_len_s1', 'name_len_cand', 'name_len_diff',
    'name_token_count_s1', 'name_token_count_cand', 'name_token_count_diff',
    'addr_len_diff', 'addr_digit_overlap',
    's1_name_is_non_latin', 'cand_name_is_non_latin'
}
present_flagged = [f for f in feature_cols if f in flagged_features]
print(f"Flagged features present in active model: {present_flagged} (Count: {len(present_flagged)})")


print("\n" + "=" * 80)
print("2 & 3. VALIDATION SET EVALUATION & CO-LOCATION GUARD COST ANALYSIS")
print("=" * 80)

# Load ground truth and source files
s1_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep='\t')
s2_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep='\t')
s3_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep='\t')
gt_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep='\t')

s23_all = pd.concat([s2_all, s3_all], ignore_index=True)
s23_all['business_name'] = s23_all['business_name'].fillna('')
s23_all['business_address'] = s23_all['business_address'].fillna('')
s1_all['business_name'] = s1_all['business_name'].fillna('')
s1_all['business_address'] = s1_all['business_address'].fillna('')

# Create a clean hold-out validation set of 10,000 entities (5,000 US + 5,000 India)
# Use a distinct seed (seed=999) to evaluate generalization
np.random.seed(999)
us_val = s1_all[s1_all['country'] == 'US'].sample(n=5000, random_state=999)
india_val = s1_all[s1_all['country'] == 'India'].sample(n=5000, random_state=999)
val_s1 = pd.concat([us_val, india_val], ignore_index=True).reset_index(drop=True)

val_s1_ids = set(val_s1['entity_id'])
gt_map = {}
for _, row in gt_all[gt_all['source1_entity_id'].isin(val_s1_ids)].iterrows():
    sid = row['source1_entity_id']
    m = row['matched_entity_ids']
    if pd.isna(m) or str(m).strip() == '':
        gt_map[sid] = set()
    else:
        gt_map[sid] = set(str(m).split(','))

# Build candidate generator matching the production pipeline
def run_prod_blocking(s1_df, s23_df, max_cands=25):
    s23_names = s23_df['business_name'].values
    s23_addrs = s23_df['business_address'].values
    s23_ids = s23_df['entity_id'].values
    s23_countries = s23_df['country'].values
    
    s23_names_norm = [normalize_name(x) for x in s23_names]
    s23_addrs_norm = [normalize_address(x) for x in s23_addrs]
    
    # Group by country
    country_indices = defaultdict(list)
    name_idx = defaultdict(lambda: defaultdict(list))
    addr_idx = defaultdict(lambda: defaultdict(list))
    
    for i in range(len(s23_df)):
        c = s23_countries[i]
        country_indices[c].append(i)
        for p in s23_names_norm[i].split():
            if len(p) >= 3:
                name_idx[c][p].append(i)
        for t in s23_addrs_norm[i].split():
            if len(t) >= 4:
                addr_idx[c][t].append(i)
                
    s1_names_norm = [normalize_name(x) for x in s1_df['business_name'].values]
    s1_addrs_norm = [normalize_address(x) for x in s1_df['business_address'].values]
    s1_ids = s1_df['entity_id'].values
    s1_countries = s1_df['country'].values
    
    cands_dict = {}
    for i in range(len(s1_df)):
        sid = s1_ids[i]
        sc = s1_countries[i]
        cands = set()
        for p in s1_names_norm[i].split():
            if len(p) >= 3 and p in name_idx[sc] and len(name_idx[sc][p]) <= 1500:
                cands.update(name_idx[sc][p])
        if len(cands) < 5:
            for t in s1_addrs_norm[i].split():
                if len(t) >= 4 and t in addr_idx[sc] and len(addr_idx[sc][t]) <= 500:
                    cands.update(addr_idx[sc][t])
                    if len(cands) >= max_cands:
                        break
        # Ranking/capping
        if len(cands) > max_cands:
            s1_n_toks = set(s1_names_norm[i].split())
            s1_a_toks = set(s1_addrs_norm[i].split())
            scored = []
            for ci in cands:
                score = 0.0
                if s1_names_norm[i] == s23_names_norm[ci] and s1_names_norm[i]:
                    score += 10.0
                c_n_toks = set(s23_names_norm[ci].split())
                if s1_n_toks and c_n_toks:
                    score += 3.0 * (len(s1_n_toks & c_n_toks) / len(s1_n_toks | c_n_toks))
                c_a_toks = set(s23_addrs_norm[ci].split())
                if s1_a_toks and c_a_toks:
                    score += 2.0 * (len(s1_a_toks & c_a_toks) / len(s1_a_toks | c_a_toks))
                scored.append((score, ci))
            scored.sort(key=lambda x: x[0], reverse=True)
            cands = [ci for _, ci in scored[:max_cands]]
            
        cands_dict[sid] = [s23_ids[ci] for ci in cands]
    return cands_dict

print("Running blocking on 10,000 validation entities...")
t0 = time.time()
val_cands = run_prod_blocking(val_s1, s23_all)
print(f"Blocking finished in {time.time()-t0:.1f}s")

# Extract features
print("Extracting features using the invariant pipeline...")
t0 = time.time()
val_feats_df = compute_features_batch(val_s1, s23_all, val_cands)
print(f"Feature computation on {len(val_feats_df)} pairs finished in {time.time()-t0:.1f}s")

# Score with loaded model
raw_scores = predict_scores(model, val_feats_df, feature_cols)

# Extract name similarity metrics for guard analysis
name_sort = val_feats_df['name_token_sort_ratio'].values
name_lev = val_feats_df['name_levenshtein'].values
name_set = val_feats_df['name_token_set_ratio'].values
max_name_sim = np.maximum(np.maximum(name_sort, name_lev), name_set)

s1_arr = val_feats_df['s1_id'].values
cand_arr = val_feats_df['cand_id'].values

# True match lookup
true_pairs_set = set()
for sid, c_set in gt_map.items():
    for cid in c_set:
        true_pairs_set.add((sid, cid))

THRESHOLD = 0.940

# Analyze the exact impact of the guard on Candidate Pairs
# Candidates passing raw threshold (score >= 0.940)
raw_passed_indices = [i for i in range(len(raw_scores)) if raw_scores[i] >= THRESHOLD]
guard_rejected_indices = [i for i in raw_passed_indices if max_name_sim[i] < 0.60]
guard_passed_indices = [i for i in raw_passed_indices if max_name_sim[i] >= 0.60]

true_matches_rejected = 0
false_positives_rejected = 0

for i in guard_rejected_indices:
    pair = (s1_arr[i], cand_arr[i])
    if pair in true_pairs_set:
        true_matches_rejected += 1
    else:
        false_positives_rejected += 1

print("\n" + "=" * 80)
print("CO-LOCATION GUARD DETAILED COST/BENEFIT BREAKDOWN (Threshold=0.940)")
print("=" * 80)
print(f"Total pairs scoring >= {THRESHOLD}: {len(raw_passed_indices)}")
print(f"Total pairs rejected by guard (max_name_sim < 0.60): {len(guard_rejected_indices)}")
print(f"  - True Matches Lost (Recall Cost / False Negatives added): {true_matches_rejected}")
print(f"  - False Positives Eliminated (Precision Gain):           {false_positives_rejected}")

if true_matches_rejected > 0:
    loss_ratio = false_positives_rejected / true_matches_rejected
    print(f"  - Precision-to-Recall trade ratio: {loss_ratio:.2f} FP removed per 1 TP lost")
else:
    print("  - True Matches Lost: 0 (Zero recall cost!)")

# Compute Full Evaluation Metrics: WITHOUT Guard vs WITH Guard
def compute_macro_metrics(passed_indices, s1_id_list, gt_dict):
    preds = defaultdict(set)
    for i in passed_indices:
        preds[s1_arr[i]].add(cand_arr[i])
        
    f05_scores = []
    prec_list = []
    rec_list = []
    tp_total = fp_total = fn_total = 0
    singleton_total = singleton_correct = 0
    
    for sid in s1_id_list:
        true_m = gt_dict.get(sid, set())
        pred_m = preds.get(sid, set())
        
        if not true_m:
            singleton_total += 1
            if not pred_m:
                singleton_correct += 1
                f05_scores.append(1.0)
                prec_list.append(1.0)
                rec_list.append(1.0)
            else:
                f05_scores.append(0.0)
                prec_list.append(0.0)
                rec_list.append(1.0)
                fp_total += len(pred_m)
        else:
            tp = len(true_m & pred_m)
            fp = len(pred_m - true_m)
            fn = len(true_m - pred_m)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05 = compute_f05(p, r)
            
            f05_scores.append(f05)
            prec_list.append(p)
            rec_list.append(r)
            
    macro_f05 = np.mean(f05_scores)
    macro_p = np.mean(prec_list)
    macro_r = np.mean(rec_list)
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    
    return {
        'macro_f05': macro_f05,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'singleton_correct': singleton_correct,
        'singleton_total': singleton_total
    }

val_s1_id_list = val_s1['entity_id'].tolist()
metrics_without_guard = compute_macro_metrics(raw_passed_indices, val_s1_id_list, gt_map)
metrics_with_guard = compute_macro_metrics(guard_passed_indices, val_s1_id_list, gt_map)

print("\n" + "=" * 80)
print("IN-DOMAIN (US/India) FRESH VALIDATION RESULTS RECOMPUTED FROM SCRATCH")
print("=" * 80)

print(f"{'Metric':<25} | {'Without Guard':<18} | {'WITH Guard (Final)':<18} | {'Delta':<10}")
print("-" * 75)
for k in ['macro_f05', 'macro_precision', 'macro_recall', 'micro_precision', 'micro_recall']:
    v1 = metrics_without_guard[k]
    v2 = metrics_with_guard[k]
    print(f"{k:<25} | {v1:<18.4f} | {v2:<18.4f} | {v2-v1:+10.4f}")

print("-" * 75)
print(f"Total True Positives (TP):   {metrics_with_guard['tp']} (vs {metrics_without_guard['tp']})")
print(f"Total False Positives (FP):  {metrics_with_guard['fp']} (vs {metrics_without_guard['fp']})")
print(f"Total False Negatives (FN):  {metrics_with_guard['fn']} (vs {metrics_without_guard['fn']})")
print(f"Singletons Correct:          {metrics_with_guard['singleton_correct']}/{metrics_with_guard['singleton_total']} ({metrics_with_guard['singleton_correct']/metrics_with_guard['singleton_total']*100:.2f}%)")

# Country breakdown:
us_ids = val_s1[val_s1['country'] == 'US']['entity_id'].tolist()
india_ids = val_s1[val_s1['country'] == 'India']['entity_id'].tolist()

us_res = compute_macro_metrics(guard_passed_indices, us_ids, gt_map)
india_res = compute_macro_metrics(guard_passed_indices, india_ids, gt_map)

print("\n" + "=" * 80)
print("COUNTRY-WISE VALIDATION METRICS (FINAL PIPELINE)")
print("=" * 80)
print(f"US (5,000 entities):    Macro F_0.5 = {us_res['macro_f05']:.4f} (Prec: {us_res['macro_precision']:.4f}, Rec: {us_res['macro_recall']:.4f})")
print(f"India (5,000 entities): Macro F_0.5 = {india_res['macro_f05']:.4f} (Prec: {india_res['macro_precision']:.4f}, Rec: {india_res['macro_recall']:.4f})")
print(f"Overall In-Domain:      Macro F_0.5 = {metrics_with_guard['macro_f05']:.4f}")

