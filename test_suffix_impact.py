#!/usr/bin/env python3
"""
Test the impact of Universal Legal-Suffix and Accent Normalization:
1. Validation on US/India holdout (10,000 entities) to confirm no regression.
2. Zero-shot US -> India proxy transfer to check precision & F_0.5 change.
"""

import os, sys, time, gc, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from normalize import normalize_name, normalize_address
from features import compute_pair_features, compute_features_batch
from model import load_model, predict_scores, compute_f05, prepare_training_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "models", "lgbm_model.pkl")

print("=" * 80)
print("TEST 1: US/INDIA IN-DOMAIN VALIDATION (CONFIRM NO REGRESSION)")
print("=" * 80)

# Load data
s1_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep='\t')
s2_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep='\t')
s3_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep='\t')
gt_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep='\t')

s23_all = pd.concat([s2_all, s3_all], ignore_index=True)
s23_all['business_name'] = s23_all['business_name'].fillna('')
s23_all['business_address'] = s23_all['business_address'].fillna('')
s1_all['business_name'] = s1_all['business_name'].fillna('')
s1_all['business_address'] = s1_all['business_address'].fillna('')

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

# Load model
model, feature_cols, base_thresh = load_model(MODEL_PATH)

def run_prod_blocking(s1_df, s23_df, max_cands=25):
    s23_names = s23_df['business_name'].values
    s23_addrs = s23_df['business_address'].values
    s23_ids = s23_df['entity_id'].values
    s23_countries = s23_df['country'].values
    
    s23_names_norm = [normalize_name(x) for x in s23_names]
    s23_addrs_norm = [normalize_address(x) for x in s23_addrs]
    
    name_idx = defaultdict(lambda: defaultdict(list))
    addr_idx = defaultdict(lambda: defaultdict(list))
    
    for i in range(len(s23_df)):
        c = s23_countries[i]
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

print("Running validation blocking & feature extraction...")
val_cands = run_prod_blocking(val_s1, s23_all)
val_feats_df = compute_features_batch(val_s1, s23_all, val_cands)

scores = predict_scores(model, val_feats_df, feature_cols)

def eval_macro(scores, feats_df, s1_id_list, gt_dict, thresh=0.940):
    s1_arr = feats_df['s1_id'].values
    cand_arr = feats_df['cand_id'].values
    preds = defaultdict(set)
    for i in range(len(scores)):
        if scores[i] >= thresh:
            preds[s1_arr[i]].add(cand_arr[i])
            
    f05_scores = []
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
            else:
                f05_scores.append(0.0)
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
            
    macro_f05 = np.mean(f05_scores)
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    
    return macro_f05, micro_p, micro_r, tp_total, fp_total, fn_total, singleton_correct, singleton_total

val_s1_id_list = val_s1['entity_id'].tolist()
f05_val, p_val, r_val, tp_v, fp_v, fn_v, sc_v, st_v = eval_macro(scores, val_feats_df, val_s1_id_list, gt_map, thresh=0.940)

print(f"Validation Macro F0.5: {f05_val:.4f}")
print(f"Validation Micro Precision: {p_val:.4f}")
print(f"Validation Micro Recall: {r_val:.4f}")
print(f"Singletons: {sc_v}/{st_v} ({sc_v/st_v*100:.2f}%)")

print("\n" + "=" * 80)
print("TEST 2: ZERO-SHOT TRANSFER (TRAIN US -> EVAL INDIA PROXY)")
print("=" * 80)

# Train on US, test on India with new normalization
us_s1_sample = s1_all[s1_all['country'] == 'US'].sample(n=6000, random_state=42).reset_index(drop=True)
india_s1_sample = s1_all[s1_all['country'] == 'India'].sample(n=4000, random_state=42).reset_index(drop=True)
us_s23 = s23_all[s23_all['country'] == 'US'].reset_index(drop=True)
india_s23 = s23_all[s23_all['country'] == 'India'].reset_index(drop=True)

us_cands = run_prod_blocking(us_s1_sample, us_s23)
india_cands = run_prod_blocking(india_s1_sample, india_s23)

us_feats = compute_features_batch(us_s1_sample, us_s23, us_cands)
india_feats = compute_features_batch(india_s1_sample, india_s23, india_cands)

us_labeled = prepare_training_data(us_feats, gt_all)
india_labeled = prepare_training_data(india_feats, gt_all)

y_us = us_labeled['label'].values
scale_pos = (len(y_us) - y_us.sum()) / y_us.sum()

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': scale_pos,
    'verbose': -1,
    'n_jobs': -1,
    'min_child_samples': 50,
    'max_depth': 6,
}

m_transfer = lgb.train(params, lgb.Dataset(us_labeled[feature_cols].values, label=y_us), num_boost_round=300)
scores_transfer = m_transfer.predict(india_labeled[feature_cols].values)

india_gt_map = {}
for _, row in gt_all[gt_all['source1_entity_id'].isin(set(india_s1_sample['entity_id']))].iterrows():
    sid = row['source1_entity_id']
    m = row['matched_entity_ids']
    if pd.isna(m) or str(m).strip() == '':
        india_gt_map[sid] = set()
    else:
        india_gt_map[sid] = set(str(m).split(','))

india_s1_ids = india_s1_sample['entity_id'].tolist()

print(f"{'Thresh':<8} | {'Macro F0.5':<12} {'Micro Prec':<12} {'Micro Rec':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
print("-" * 75)
for t in [0.70, 0.80, 0.85, 0.90, 0.94, 0.97, 0.98]:
    f, p, r, tp, fp, fn, _, _ = eval_macro(scores_transfer, india_labeled, india_s1_ids, india_gt_map, thresh=t)
    print(f"{t:<8.2f} | {f:<12.4f} {p:<12.4f} {r:<12.4f} {tp:<6} {fp:<6} {fn:<6}")

