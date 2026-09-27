#!/usr/bin/env python3
"""
DIAGNOSIS & METRIC RECONCILIATION SCRIPT:
1. Side-by-side composition of both validation sets (curated 50k pool vs full 8.5M pool).
2. Re-run exact audit.py split to verify original 0.974 score.
3. Bisect classifier threshold vs blocking recall on the full S23 database.
4. Measure exact effect of suffix/accent normalization on a unified benchmark.
"""

import os, sys, time, gc, pickle
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from normalize import normalize_name, normalize_address
from features import compute_pair_features, compute_features_batch
from model import load_model, predict_scores, compute_f05, prepare_training_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "models", "lgbm_model.pkl")

# Load ground truth and sources
print("Loading data...")
s1_full = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t")
s2_full = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t")
s3_full = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t")
gt_full = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")

for df in [s1_full, s2_full, s3_full]:
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')

s23_full = pd.concat([s2_full, s3_full], ignore_index=True)
del s2_full, s3_full; gc.collect()

print("=" * 80)
print("INVESTIGATION 1: COMPOSITION OF BOTH VALIDATION SETS")
print("=" * 80)

# Set A: audit.py / quick_test.py setup
SAMPLE_FRACTION = 0.005
np.random.seed(42)
n_sample = int(len(s1_full) * SAMPLE_FRACTION)
s1_sample_A = s1_full.sample(n_sample, random_state=42).reset_index(drop=True)
gt_sample_A = gt_full[gt_full['source1_entity_id'].isin(s1_sample_A['entity_id'])].reset_index(drop=True)

all_matched_ids_A = set()
for _, row in gt_sample_A.iterrows():
    if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
        all_matched_ids_A.update(str(row['matched_entity_ids']).split(','))

matched_mask_A = s23_full['entity_id'].isin(all_matched_ids_A)
s23_matched_A = s23_full[matched_mask_A]
s23_unmatched_A = s23_full[~matched_mask_A]
sample_countries_A = set(s1_sample_A['country'].unique())
s23_unmatched_A = s23_unmatched_A[s23_unmatched_A['country'].isin(sample_countries_A)]
n_unmatched_sample_A = min(int(len(s23_unmatched_A) * SAMPLE_FRACTION * 5), 50000)
s23_unmatched_sample_A = s23_unmatched_A.sample(n_unmatched_sample_A, random_state=42)
s23_sample_A = pd.concat([s23_matched_A, s23_unmatched_sample_A], ignore_index=True)

val_frac_A = 0.3
val_n_A = int(len(s1_sample_A) * val_frac_A)
val_indices_A = np.random.choice(len(s1_sample_A), val_n_A, replace=False)
val_mask_A = np.zeros(len(s1_sample_A), dtype=bool)
val_mask_A[val_indices_A] = True
val_s1_A = s1_sample_A[val_mask_A].reset_index(drop=True)
val_gt_A = gt_sample_A[gt_sample_A['source1_entity_id'].isin(val_s1_A['entity_id'])].reset_index(drop=True)

# Set B: verify_final.py setup
np.random.seed(999)
us_val_B = s1_full[s1_full['country'] == 'US'].sample(n=5000, random_state=999)
india_val_B = s1_full[s1_full['country'] == 'India'].sample(n=5000, random_state=999)
val_s1_B = pd.concat([us_val_B, india_val_B], ignore_index=True).reset_index(drop=True)
val_gt_B = gt_full[gt_full['source1_entity_id'].isin(val_s1_B['entity_id'])].reset_index(drop=True)

def analyze_set(name, s1_df, gt_df, s23_df):
    gt_map = {}
    for _, row in gt_df.iterrows():
        sid = row['source1_entity_id']
        m = row['matched_entity_ids']
        if pd.isna(m) or str(m).strip() == '':
            gt_map[sid] = set()
        else:
            gt_map[sid] = set(str(m).split(','))
    singletons = sum(1 for v in gt_map.values() if len(v) == 0)
    matched = len(gt_map) - singletons
    total_true_pairs = sum(len(v) for v in gt_map.values())
    print(f"{name}:")
    print(f"  S1 Entities: {len(s1_df)} ({matched} with matches, {singletons} singletons [singleton %: {singletons/len(s1_df)*100:.1f}%])")
    print(f"  Total Ground Truth Matches: {total_true_pairs}")
    print(f"  Target S23 Candidate Pool Size: {len(s23_df):,} records")

analyze_set("SET A (Original audit.py 0.974 setup)", val_s1_A, val_gt_A, s23_sample_A)
print()
analyze_set("SET B (verify_final.py 0.7247 setup)", val_s1_B, val_gt_B, s23_full)

print("\n" + "=" * 80)
print("INVESTIGATION 2: RE-RUNNING ON SET A (EXACT AUDIT SETUP)")
print("=" * 80)

from blocking import run_blocking

cands_A = run_blocking(val_s1_A, s23_sample_A, use_tfidf=True, n_neighbors=10)
feats_A = compute_features_batch(val_s1_A, s23_sample_A, cands_A)
model, feature_cols, base_thresh = load_model(MODEL_PATH)
scores_A = predict_scores(model, feats_A, feature_cols)

def evaluate_gt_map(scores, feats_df, s1_list, gt_df, thresh):
    gt_map = {}
    for _, row in gt_df.iterrows():
        sid = row['source1_entity_id']
        m = row['matched_entity_ids']
        if pd.isna(m) or str(m).strip() == '':
            gt_map[sid] = set()
        else:
            gt_map[sid] = set(str(m).split(','))
            
    s1_arr = feats_df['s1_id'].values
    cand_arr = feats_df['cand_id'].values
    preds = defaultdict(set)
    for i in range(len(scores)):
        if scores[i] >= thresh:
            preds[s1_arr[i]].add(cand_arr[i])
            
    f05_list, p_list, r_list = [], [], []
    tp_tot = fp_tot = fn_tot = 0
    singletons_correct = singletons_total = 0
    
    for sid in s1_list:
        true_m = gt_map.get(sid, set())
        pred_m = preds.get(sid, set())
        if not true_m:
            singletons_total += 1
            if not pred_m:
                singletons_correct += 1
                f05_list.append(1.0); p_list.append(1.0); r_list.append(1.0)
            else:
                f05_list.append(0.0); p_list.append(0.0); r_list.append(1.0)
                fp_tot += len(pred_m)
        else:
            tp = len(true_m & pred_m)
            fp = len(pred_m - true_m)
            fn = len(true_m - pred_m)
            tp_tot += tp; fp_tot += fp; fn_tot += fn
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05_list.append(compute_f05(p, r))
            p_list.append(p); r_list.append(r)
            
    return np.mean(f05_list), np.mean(p_list), np.mean(r_list), tp_tot, fp_tot, fn_tot, singletons_correct, singletons_total

s1_list_A = val_s1_A['entity_id'].tolist()
print("\nThreshold Sweep on SET A:")
print(f"{'Thresh':<8} | {'Macro F0.5':<12} {'Macro Prec':<12} {'Macro Rec':<12} {'Micro Prec':<12} {'Micro Rec':<12} {'Singletons'}")
print("-" * 80)
for t in [0.50, 0.70, 0.80, 0.90, 0.94]:
    f, p, r, tp, fp, fn, sc, st = evaluate_gt_map(scores_A, feats_A, s1_list_A, val_gt_A, t)
    mp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"{t:<8.2f} | {f:<12.4f} {p:<12.4f} {r:<12.4f} {mp:<12.4f} {mr:<12.4f} ({sc}/{st})")

print("\n" + "=" * 80)
print("INVESTIGATION 3: THRESHOLD & BLOCKING ANALYSIS ON SET B (FULL S23 POOL)")
print("=" * 80)

# Blocking recall on Set B
from test_suffix_impact import run_prod_blocking

cands_B = run_prod_blocking(val_s1_B, s23_full)

gt_map_B = {}
for _, row in val_gt_B.iterrows():
    sid = row['source1_entity_id']
    m = row['matched_entity_ids']
    if pd.isna(m) or str(m).strip() == '':
        gt_map_B[sid] = set()
    else:
        gt_map_B[sid] = set(str(m).split(','))

total_true_B = sum(len(v) for v in gt_map_B.values())
found_in_cands_B = sum(len(gt_map_B[sid] & set(cands_B.get(sid, []))) for sid in gt_map_B)
print(f"Set B Blocking Recall on full 8.5M pool: {found_in_cands_B}/{total_true_B} = {found_in_cands_B/total_true_B*100:.2f}%")

feats_B = compute_features_batch(val_s1_B, s23_full, cands_B)
scores_B = predict_scores(model, feats_B, feature_cols)
s1_list_B = val_s1_B['entity_id'].tolist()

print("\nThreshold Sweep on SET B (Full 8.5M Pool):")
print(f"{'Thresh':<8} | {'Macro F0.5':<12} {'Macro Prec':<12} {'Macro Rec':<12} {'Micro Prec':<12} {'Micro Rec':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
print("-" * 85)
for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.94]:
    f, p, r, tp, fp, fn, sc, st = evaluate_gt_map(scores_B, feats_B, s1_list_B, val_gt_B, t)
    mp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"{t:<8.2f} | {f:<12.4f} {p:<12.4f} {r:<12.4f} {mp:<12.4f} {mr:<12.4f} {tp:<6} {fp:<6} {fn:<6}")

