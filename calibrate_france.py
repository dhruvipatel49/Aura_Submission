#!/usr/bin/env python3
"""
CALIBRATION SCRIPT:
1. Feature set ablation: Country-invariant features vs raw length features
2. US -> India zero-shot experiment with feature sets and threshold sweep
3. Singleton false-positive failure analysis & guard design
4. Calibrate France-specific threshold and test-set statistics
"""

import os, sys, time, gc
import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from normalize import normalize_name, normalize_address, extract_digits
from features import compute_features_batch
from model import prepare_training_data, compute_f05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

print("=" * 80)
print("SECTION 1 & 2: FEATURE IMPORTANCES, COUNTRY INVARIANCE & THRESHOLD SWEEP")
print("=" * 80)

# Load sample
s1_train = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep='\t')
s2_train = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep='\t')
s3_train = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep='\t')
gt_train = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep='\t')

s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
s23_train['business_name'] = s23_train['business_name'].fillna('')
s23_train['business_address'] = s23_train['business_address'].fillna('')

s1_train['business_name'] = s1_train['business_name'].fillna('')
s1_train['business_address'] = s1_train['business_address'].fillna('')

us_s1 = s1_train[s1_train['country'] == 'US'].sample(n=6000, random_state=42).reset_index(drop=True)
india_s1 = s1_train[s1_train['country'] == 'India'].sample(n=4000, random_state=42).reset_index(drop=True)

us_s23 = s23_train[s23_train['country'] == 'US'].reset_index(drop=True)
india_s23 = s23_train[s23_train['country'] == 'India'].reset_index(drop=True)

def build_blocking_cands(s1_df, s23_df, max_cands=25):
    s23_names = s23_df['business_name'].values
    s23_addrs = s23_df['business_address'].values
    s23_ids = s23_df['entity_id'].values
    s23_names_norm = [normalize_name(x) for x in s23_names]
    s23_addrs_norm = [normalize_address(x) for x in s23_addrs]
    
    name_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    for i in range(len(s23_df)):
        parts = s23_names_norm[i].split()
        for p in parts:
            if len(p) >= 3:
                name_idx[p].append(i)
        for t in s23_addrs_norm[i].split():
            if len(t) >= 4:
                addr_idx[t].append(i)
                
    s1_names_norm = [normalize_name(x) for x in s1_df['business_name'].values]
    s1_addrs_norm = [normalize_address(x) for x in s1_df['business_address'].values]
    s1_ids = s1_df['entity_id'].values
    
    cands_dict = {}
    for i in range(len(s1_df)):
        sid = s1_ids[i]
        cands = set()
        for p in s1_names_norm[i].split():
            if len(p) >= 3 and p in name_idx and len(name_idx[p]) <= 1500:
                cands.update(name_idx[p])
        if len(cands) < 5:
            for t in s1_addrs_norm[i].split():
                if len(t) >= 4 and t in addr_idx and len(addr_idx[t]) <= 500:
                    cands.update(addr_idx[t])
                    if len(cands) >= max_cands:
                        break
        # Ranking/capping
        if len(cands) > max_cands:
            s1_n_toks = set(s1_names_norm[i].split())
            s1_a_toks = set(s1_addrs_norm[i].split())
            scored = []
            for ci in cands:
                sc = 0.0
                if s1_names_norm[i] == s23_names_norm[ci] and s1_names_norm[i]:
                    sc += 10.0
                c_n_toks = set(s23_names_norm[ci].split())
                if s1_n_toks and c_n_toks:
                    sc += 3.0 * (len(s1_n_toks & c_n_toks) / len(s1_n_toks | c_n_toks))
                c_a_toks = set(s23_addrs_norm[ci].split())
                if s1_a_toks and c_a_toks:
                    sc += 2.0 * (len(s1_a_toks & c_a_toks) / len(s1_a_toks | c_a_toks))
                scored.append((sc, ci))
            scored.sort(key=lambda x: x[0], reverse=True)
            cands = [ci for _, ci in scored[:max_cands]]
            
        cands_dict[sid] = [s23_ids[ci] for ci in cands]
    return cands_dict

print("Generating candidates...")
us_cands = build_blocking_cands(us_s1, us_s23)
india_cands = build_blocking_cands(india_s1, india_s23)

print("Extracting features...")
us_feats_df = compute_features_batch(us_s1, us_s23, us_cands)
india_feats_df = compute_features_batch(india_s1, india_s23, india_cands)

us_labeled = prepare_training_data(us_feats_df, gt_train)
india_labeled = prepare_training_data(india_feats_df, gt_train)

all_features = [c for c in us_feats_df.columns if c not in {'s1_id', 'cand_id', 'label'}]
country_specific_features = {
    'name_len_s1', 'name_len_cand', 'name_len_diff',
    'name_token_count_s1', 'name_token_count_cand', 'name_token_count_diff',
    'addr_len_diff', 'addr_digit_overlap',
    's1_name_is_non_latin', 'cand_name_is_non_latin'
}
invariant_features = [f for f in all_features if f not in country_specific_features]

y_us = us_labeled['label'].values
scale_pos_weight = (len(y_us) - y_us.sum()) / y_us.sum()

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'n_jobs': -1,
    'min_child_samples': 50,
    'max_depth': 6,
}

model_all = lgb.train(params, lgb.Dataset(us_labeled[all_features].values, label=y_us), num_boost_round=300)
model_inv = lgb.train(params, lgb.Dataset(us_labeled[invariant_features].values, label=y_us), num_boost_round=300)

scores_india_all = model_all.predict(india_labeled[all_features].values)
scores_india_inv = model_inv.predict(india_labeled[invariant_features].values)

# Evaluate macro F0.5
def evaluate_predictions(scores, feats_df, s1_ids_list, gt_df, thresh):
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids']
        if pd.isna(matched) or str(matched).strip() == '':
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = set(str(matched).split(','))
            
    entity_preds = defaultdict(set)
    s1_ids = feats_df['s1_id'].values
    cands = feats_df['cand_id'].values
    for i in range(len(scores)):
        if scores[i] >= thresh:
            entity_preds[s1_ids[i]].add(cands[i])
            
    f05_list = []
    tp_tot = fp_tot = fn_tot = 0
    singleton_tot = singleton_correct = 0
    
    for sid in s1_ids_list:
        true_m = gt_map.get(sid, set())
        pred_m = entity_preds.get(sid, set())
        if not true_m:
            singleton_tot += 1
            if not pred_m:
                singleton_correct += 1
                f05_list.append(1.0)
            else:
                f05_list.append(0.0)
                fp_tot += len(pred_m)
        else:
            tp = len(true_m & pred_m)
            fp = len(pred_m - true_m)
            fn = len(true_m - pred_m)
            tp_tot += tp
            fp_tot += fp
            fn_tot += fn
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f05_list.append(compute_f05(p, r))
            
    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0
    rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0
    macro_f05 = np.mean(f05_list)
    return macro_f05, prec, rec, singleton_correct, singleton_tot, tp_tot, fp_tot, fn_tot

print("\n" + "=" * 80)
print("THRESHOLD SWEEP ON ZERO-SHOT INDIA TRANSFER")
print("=" * 80)
print(f"{'Thresh':<8} | {'All: F0.5':<10} {'Prec':<8} {'Rec':<8} | {'Inv: F0.5':<10} {'Prec':<8} {'Rec':<8}")
print("-" * 75)

s1_india_ids = india_s1['entity_id'].tolist()
for t in [0.5, 0.7, 0.8, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
    f_all, p_all, r_all, _, _, _, _, _ = evaluate_predictions(scores_india_all, india_labeled, s1_india_ids, gt_train, t)
    f_inv, p_inv, r_inv, sc, st, tp, fp, fn = evaluate_predictions(scores_india_inv, india_labeled, s1_india_ids, gt_train, t)
    print(f"{t:<8.3f} | {f_all:<10.4f} {p_all:<8.4f} {r_all:<8.4f} | {f_inv:<10.4f} {p_inv:<8.4f} {r_inv:<8.4f}")

# Feature importances
imp_all = sorted(zip(all_features, model_all.feature_importance(importance_type='gain')), key=lambda x: -x[1])
imp_inv = sorted(zip(invariant_features, model_inv.feature_importance(importance_type='gain')), key=lambda x: -x[1])

print("\nTOP FEATURES IN ORIGINAL MODEL (with flagged country-specific features):")
for f, g in imp_all[:15]:
    flag = " [FLAGGED COUNTRY-SPECIFIC]" if f in country_specific_features else ""
    print(f"  {f:<28}: {g:>14.1f}{flag}")

print("\nTOP FEATURES IN INVARIANT MODEL:")
for f, g in imp_inv[:15]:
    print(f"  {f:<28}: {g:>14.1f}")

# SECTION 5: SINGLETON FP INVESTIGATION
print("\n" + "=" * 80)
print("SECTION 5: SINGLETON FALSE POSITIVE CASE STUDY & GUARD ANALYSIS")
print("=" * 80)

gt_map = {row['source1_entity_id']: set(str(row['matched_entity_ids']).split(',')) if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip() else set()
          for _, row in gt_train.iterrows()}

india_labeled['score_inv'] = scores_india_inv

s1_dict = {r['entity_id']: (r['business_name'], r['business_address']) for _, r in india_s1.iterrows()}
s23_dict = {r['entity_id']: (r['business_name'], r['business_address']) for _, r in india_s23.iterrows()}

fp_cases = []
s1_arr = india_labeled['s1_id'].values
cand_arr = india_labeled['cand_id'].values
score_arr = india_labeled['score_inv'].values
name_sort_arr = india_labeled['name_token_sort_ratio'].values
addr_sort_arr = india_labeled['addr_token_sort_ratio'].values
addr_jacc_arr = india_labeled['addr_token_jaccard'].values
max_addr_sim_arr = india_labeled['max_addr_sim'].values
both_addr_arr = india_labeled['both_have_addr'].values

for i in range(len(india_labeled)):
    sid = s1_arr[i]
    if len(gt_map.get(sid, set())) == 0 and score_arr[i] >= 0.95:
        cid = cand_arr[i]
        s1_n, s1_a = s1_dict.get(sid, ('', ''))
        c_n, c_a = s23_dict.get(cid, ('', ''))
        fp_cases.append({
            's1_id': sid, 'cand_id': cid, 'score': score_arr[i],
            's1_name': s1_n, 'cand_name': c_n,
            's1_addr': s1_a, 'cand_addr': c_a,
            'name_sort': name_sort_arr[i],
            'addr_sort': addr_sort_arr[i],
            'addr_jaccard': addr_jacc_arr[i],
            'max_addr_sim': max_addr_sim_arr[i],
            'both_addr': both_addr_arr[i]
        })

print(f"Total FP singletons at threshold >= 0.95: {len(fp_cases)}")
for idx, c in enumerate(fp_cases[:8]):
    print(f"\nCase {idx+1}:")
    print(f"  S1: '{c['s1_name']}' | Address: '{c['s1_addr']}'")
    print(f"  C:  '{c['cand_name']}' | Address: '{c['cand_addr']}'")
    print(f"  Score={c['score']:.4f} | NameSortRatio={c['name_sort']:.2f} | AddrSortRatio={c['addr_sort']:.2f} | AddrJaccard={c['addr_jaccard']:.2f}")

# Test guard rule:
# If both records have addresses, require at least some address similarity (e.g. max_addr_sim >= 0.15 or addr_jaccard > 0)
# or if name is very short (len < 5), require high name sort ratio
print("\n" + "=" * 80)
print("TESTING SINGLETON GUARD RULE IMPACT ON ZERO-SHOT INDIA")
print("=" * 80)

def apply_guard(row_score, both_addr, max_addr_sim, name_sort):
    # If both have addresses, but address similarity is virtually 0, reject false positive matches
    if both_addr == 1.0 and max_addr_sim < 0.15:
        return 0.0
    return row_score

guarded_scores = np.array([
    apply_guard(score_arr[i], both_addr_arr[i], max_addr_sim_arr[i], name_sort_arr[i])
    for i in range(len(score_arr))
])

for t in [0.90, 0.94, 0.96, 0.97, 0.98]:
    f_raw, p_raw, r_raw, _, _, _, _, _ = evaluate_predictions(score_arr, india_labeled, s1_india_ids, gt_train, t)
    f_grd, p_grd, r_grd, sc, st, tp, fp, fn = evaluate_predictions(guarded_scores, india_labeled, s1_india_ids, gt_train, t)
    print(f"Thresh {t:.2f} | Raw: F0.5={f_raw:.4f}, Prec={p_raw:.4f}, Rec={r_raw:.4f} | Guarded: F0.5={f_grd:.4f}, Prec={p_grd:.4f}, Rec={r_grd:.4f} (Singletons {sc}/{st})")

print("\nCALIBRATION EXPERIMENT DONE.")
