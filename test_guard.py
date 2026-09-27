#!/usr/bin/env python3
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from normalize import normalize_name, normalize_address
from features import compute_features_batch
from model import prepare_training_data, compute_f05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

# Load data
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

from calibrate_france import build_blocking_cands, evaluate_predictions, invariant_features

us_cands = build_blocking_cands(us_s1, us_s23)
india_cands = build_blocking_cands(india_s1, india_s23)

us_feats_df = compute_features_batch(us_s1, us_s23, us_cands)
india_feats_df = compute_features_batch(india_s1, india_s23, india_cands)

us_labeled = prepare_training_data(us_feats_df, gt_train)
india_labeled = prepare_training_data(india_feats_df, gt_train)

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

model_inv = lgb.train(params, lgb.Dataset(us_labeled[invariant_features].values, label=y_us), num_boost_round=300)
scores_india_inv = model_inv.predict(india_labeled[invariant_features].values)

# Guard against co-located distinct businesses
name_sort_arr = india_labeled['name_token_sort_ratio'].values
name_lev_arr = india_labeled['name_levenshtein'].values
name_set_arr = india_labeled['name_token_set_ratio'].values

guarded_scores = []
for i in range(len(scores_india_inv)):
    sc = scores_india_inv[i]
    # If name similarity is low, it's a co-located different business
    if max(name_sort_arr[i], name_lev_arr[i], name_set_arr[i]) < 0.65:
        guarded_scores.append(0.0)
    else:
        guarded_scores.append(sc)
guarded_scores = np.array(guarded_scores)

s1_india_ids = india_s1['entity_id'].tolist()
print("\nGUARD COMPARISON ON ZERO-SHOT INDIA TRANSFER:")
print(f"{'Thresh':<8} | {'Raw: F0.5':<10} {'Prec':<8} {'Rec':<8} | {'Guarded: F0.5':<12} {'Prec':<8} {'Rec':<8} {'Singletons'}")
print("-" * 80)
for t in [0.85, 0.90, 0.94, 0.96, 0.97, 0.98]:
    f_raw, p_raw, r_raw, _, _, _, _, _ = evaluate_predictions(scores_india_inv, india_labeled, s1_india_ids, gt_train, t)
    f_grd, p_grd, r_grd, sc, st, tp, fp, fn = evaluate_predictions(guarded_scores, india_labeled, s1_india_ids, gt_train, t)
    print(f"{t:<8.2f} | {f_raw:<10.4f} {p_raw:<8.4f} {r_raw:<8.4f} | {f_grd:<12.4f} {p_grd:<8.4f} {r_grd:<8.4f} ({sc}/{st})")

