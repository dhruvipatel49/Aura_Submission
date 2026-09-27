#!/usr/bin/env python3
"""
Quick validation run on a small sample to test the pipeline end-to-end.
Optimized for speed.
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from blocking import run_blocking, evaluate_blocking_recall
from features import compute_features_batch
from model import (
    prepare_training_data, train_model, predict_scores,
    optimize_threshold, generate_predictions, compute_f05,
    save_model
)

BASE = "/Users/daxsavaliya/Desktop/Aura_submission"
TRAIN = os.path.join(BASE, "dataset/train")

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

t0 = time.time()
s1 = pd.read_csv(os.path.join(TRAIN, "train_source1.tsv"), sep="\t")
s2 = pd.read_csv(os.path.join(TRAIN, "train_source2.tsv"), sep="\t")
s3 = pd.read_csv(os.path.join(TRAIN, "train_source3.tsv"), sep="\t")
gt = pd.read_csv(os.path.join(TRAIN, "train_ground_truth.tsv"), sep="\t")
print(f"Loaded in {time.time()-t0:.1f}s")

# Fill NaN
for df in [s1, s2, s3]:
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')

# SAMPLE: Use 0.5% of S1 and corresponding proportional subset of S2/S3 by country
SAMPLE_FRACTION = 0.005
np.random.seed(42)

# Sample S1
n_sample = int(len(s1) * SAMPLE_FRACTION)
s1_sample = s1.sample(n_sample, random_state=42).reset_index(drop=True)
gt_sample = gt[gt['source1_entity_id'].isin(s1_sample['entity_id'])].reset_index(drop=True)

# Get all matched IDs from ground truth to ensure they're in our S2/S3 subset
all_matched_ids = set()
for _, row in gt_sample.iterrows():
    if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
        all_matched_ids.update(str(row['matched_entity_ids']).split(','))

# Sample S2+S3: keep all matched IDs + a random fraction of the rest
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()

# Find which S23 entities are in matched IDs
matched_mask = s23['entity_id'].isin(all_matched_ids)
s23_matched = s23[matched_mask]

# Sample from remaining
s23_unmatched = s23[~matched_mask]
# Use countries from our S1 sample only
sample_countries = set(s1_sample['country'].unique())
s23_unmatched = s23_unmatched[s23_unmatched['country'].isin(sample_countries)]
n_unmatched_sample = min(int(len(s23_unmatched) * SAMPLE_FRACTION * 5), 50000)
s23_unmatched_sample = s23_unmatched.sample(n_unmatched_sample, random_state=42)

s23_sample = pd.concat([s23_matched, s23_unmatched_sample], ignore_index=True)
del s23, s23_unmatched, s23_matched
gc.collect()

print(f"\nSample: {len(s1_sample)} S1, {len(s23_sample)} S2+S3, {len(gt_sample)} GT")
print(f"  Matched IDs in S2+S3: {len(all_matched_ids)}")

# Split into train/val
val_frac = 0.3
val_n = int(len(s1_sample) * val_frac)
val_indices = np.random.choice(len(s1_sample), val_n, replace=False)
val_mask = np.zeros(len(s1_sample), dtype=bool)
val_mask[val_indices] = True

train_s1 = s1_sample[~val_mask].reset_index(drop=True)
val_s1 = s1_sample[val_mask].reset_index(drop=True)
train_gt = gt_sample[gt_sample['source1_entity_id'].isin(train_s1['entity_id'])].reset_index(drop=True)
val_gt = gt_sample[gt_sample['source1_entity_id'].isin(val_s1['entity_id'])].reset_index(drop=True)
print(f"Train: {len(train_s1)} S1, Val: {len(val_s1)} S1")

# ===== BLOCKING =====
print("\n" + "=" * 80)
print("STEP 1: BLOCKING (Train)")
print("=" * 80)
train_candidates = run_blocking(train_s1, s23_sample, use_tfidf=True, n_neighbors=10)
train_recall = evaluate_blocking_recall(train_candidates, train_gt)

# ===== FEATURES =====
print("\n" + "=" * 80)
print("STEP 2: FEATURES (Train)")
print("=" * 80)
train_features = compute_features_batch(train_s1, s23_sample, train_candidates)
train_features = prepare_training_data(train_features, train_gt)

# ===== MODEL =====
print("\n" + "=" * 80)
print("STEP 3: MODEL TRAINING")
print("=" * 80)
model, feature_cols = train_model(train_features)

# ===== VALIDATION =====
print("\n" + "=" * 80)
print("STEP 4: VALIDATION")
print("=" * 80)
val_candidates = run_blocking(val_s1, s23_sample, use_tfidf=True, n_neighbors=10)
val_recall = evaluate_blocking_recall(val_candidates, val_gt)
val_features = compute_features_batch(val_s1, s23_sample, val_candidates)
val_scores = predict_scores(model, val_features, feature_cols)

# Optimize threshold
val_threshold, val_f05 = optimize_threshold(val_features, val_scores, val_gt)

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"Training blocking recall: {train_recall:.4f}")
print(f"Validation blocking recall: {val_recall:.4f}")
print(f"Validation F_0.5 (macro): {val_f05:.4f}")
print(f"Optimal threshold: {val_threshold:.3f}")
print(f"Total time: {time.time()-t0:.1f}s")

# Save model artifact
model_path = os.path.join(BASE, "code/business_entity_resolution/models/lgbm_model.pkl")
save_model(model, feature_cols, val_threshold, model_path)
