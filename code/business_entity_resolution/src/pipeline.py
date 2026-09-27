#!/usr/bin/env python3
"""
Main pipeline for Business Entity Resolution.
End-to-end: blocking → features → model → thresholding → output.

Usage:
  # Train + validate on training data
  python3 pipeline.py --mode train --base-dir /path/to/Aura_submission

  # Run inference on test data
  python3 pipeline.py --mode infer --base-dir /path/to/Aura_submission

  # Full pipeline: train + infer
  python3 pipeline.py --mode full --base-dir /path/to/Aura_submission
"""

import argparse
import os
import sys
import time
import gc
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import run_blocking, evaluate_blocking_recall
from features import compute_features_batch, compute_pair_features, get_feature_cols
from model import (
    prepare_training_data, train_model, predict_scores,
    optimize_threshold, generate_predictions,
    save_predictions, save_candidate_pairs, save_model, load_model
)


def load_data(base_dir, split='train'):
    """Load source data files."""
    data_dir = os.path.join(base_dir, 'dataset', split)

    print(f"Loading {split} data from {data_dir}...")
    s1 = pd.read_csv(os.path.join(data_dir, f'{split}_source1.tsv'), sep='\t')
    s2 = pd.read_csv(os.path.join(data_dir, f'{split}_source2.tsv'), sep='\t')
    s3 = pd.read_csv(os.path.join(data_dir, f'{split}_source3.tsv'), sep='\t')

    print(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")

    # Fill NaN in text columns
    for df in [s1, s2, s3]:
        df['business_name'] = df['business_name'].fillna('')
        df['business_address'] = df['business_address'].fillna('')
        df['country'] = df['country'].fillna('')

    gt = None
    if split == 'train':
        gt_path = os.path.join(data_dir, 'train_ground_truth.tsv')
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path, sep='\t')
            print(f"  Ground truth: {len(gt)} entities")

    return s1, s2, s3, gt


def create_validation_split(s1, gt, val_fraction=0.15, seed=42):
    """
    Create a validation split from training data.
    """
    np.random.seed(seed)
    s1_ids = s1['entity_id'].values
    n_val = int(len(s1_ids) * val_fraction)

    val_indices = np.random.choice(len(s1_ids), n_val, replace=False)
    val_mask = np.zeros(len(s1_ids), dtype=bool)
    val_mask[val_indices] = True

    train_s1 = s1[~val_mask].copy()
    val_s1 = s1[val_mask].copy()

    train_gt = gt[gt['source1_entity_id'].isin(train_s1['entity_id'])].copy()
    val_gt = gt[gt['source1_entity_id'].isin(val_s1['entity_id'])].copy()

    print(f"  Train split: {len(train_s1)} S1, {len(train_gt)} GT")
    print(f"  Val split: {len(val_s1)} S1, {len(val_gt)} GT")

    return train_s1, val_s1, train_gt, val_gt


def run_pipeline_chunk(s1_df, s23_df, gt_df=None, model=None, feature_cols=None,
                       threshold=None, mode='train', use_tfidf=True, n_neighbors=10):
    """
    Run the pipeline for a chunk of S1 entities.

    In train mode: blocking → features → labels → train model
    In infer mode: blocking → features → predict
    """
    # Step 1: Blocking
    candidates = run_blocking(s1_df, s23_df, use_tfidf=use_tfidf, n_neighbors=n_neighbors)

    # Evaluate blocking recall if ground truth available
    if gt_df is not None:
        recall, missed = evaluate_blocking_recall(candidates, gt_df)

    # Step 2: Features
    features_df = compute_features_batch(s1_df, s23_df, candidates)

    if mode == 'train':
        # Step 3: Label
        features_df = prepare_training_data(features_df, gt_df)

        # Step 4: Train
        model, feature_cols = train_model(features_df)

        # Step 5: Predict on training data for threshold optimization
        scores = predict_scores(model, features_df, feature_cols)

        # Step 6: Optimize threshold
        threshold, f05 = optimize_threshold(features_df, scores, gt_df)

        return model, feature_cols, threshold, candidates, features_df, scores

    else:  # infer mode
        assert model is not None and feature_cols is not None and threshold is not None

        # Predict
        scores = predict_scores(model, features_df, feature_cols)

        return candidates, features_df, scores


def run_train(base_dir, val_fraction=0.15, sample_fraction=None):
    """
    Train the model on training data with validation split.
    """
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE")
    print("=" * 80)

    s1, s2, s3, gt = load_data(base_dir, 'train')

    # Combine S2 and S3
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined S2+S3: {len(s23)}")
    del s2, s3
    gc.collect()

    # Sample for faster iteration if requested
    if sample_fraction and sample_fraction < 1.0:
        n_sample = int(len(s1) * sample_fraction)
        print(f"\n  Sampling {n_sample} S1 entities ({sample_fraction*100:.0f}%)")
        s1_sample = s1.sample(n_sample, random_state=42)
        gt_sample = gt[gt['source1_entity_id'].isin(s1_sample['entity_id'])]
    else:
        s1_sample = s1
        gt_sample = gt

    # Validation split
    print("\n--- Creating validation split ---")
    train_s1, val_s1, train_gt, val_gt = create_validation_split(
        s1_sample, gt_sample, val_fraction=val_fraction
    )

    # Train
    print("\n--- Training ---")
    result = run_pipeline_chunk(
        train_s1, s23, train_gt, mode='train',
        use_tfidf=True, n_neighbors=10
    )
    model, feature_cols, threshold, train_candidates, train_features, train_scores = result

    # Validate
    print("\n--- Validation ---")
    val_result = run_pipeline_chunk(
        val_s1, s23, val_gt,
        model=model, feature_cols=feature_cols, threshold=threshold,
        mode='infer', use_tfidf=True, n_neighbors=10
    )
    val_candidates, val_features, val_scores = val_result

    # Evaluate on validation
    print("\n--- Validation Results ---")
    val_features_labeled = prepare_training_data(val_features, val_gt)
    val_threshold, val_f05 = optimize_threshold(val_features, val_scores, val_gt)

    # Use validation-tuned threshold
    print(f"\n  Using validation-tuned threshold: {val_threshold:.3f}")
    print(f"  Validation F_0.5: {val_f05:.4f}")

    # Save model
    model_dir = os.path.join(base_dir, 'code', 'business_entity_resolution', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lgbm_model.pkl')
    save_model(model, feature_cols, val_threshold, model_path)

    return model, feature_cols, val_threshold, val_f05


def run_train_full(base_dir, sample_fraction=None):
    """
    Retrain on FULL training data (no validation split) for final model.
    Uses threshold from validation run.
    """
    print("\n" + "=" * 80)
    print("FULL TRAINING (no val split)")
    print("=" * 80)

    s1, s2, s3, gt = load_data(base_dir, 'train')

    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined S2+S3: {len(s23)}")
    del s2, s3
    gc.collect()

    if sample_fraction and sample_fraction < 1.0:
        n_sample = int(len(s1) * sample_fraction)
        print(f"\n  Sampling {n_sample} S1 entities ({sample_fraction*100:.0f}%)")
        s1 = s1.sample(n_sample, random_state=42)
        gt = gt[gt['source1_entity_id'].isin(s1['entity_id'])]

    result = run_pipeline_chunk(
        s1, s23, gt, mode='train',
        use_tfidf=True, n_neighbors=10
    )
    model, feature_cols, threshold, _, _, _ = result

    model_dir = os.path.join(base_dir, 'code', 'business_entity_resolution', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lgbm_model_full.pkl')
    save_model(model, feature_cols, threshold, model_path)

    return model, feature_cols, threshold


def run_inference(base_dir, model=None, feature_cols=None, threshold=None,
                  model_path=None, chunk_size=50000):
    """
    Run inference on test data in chunks.
    """
    print("\n" + "=" * 80)
    print("INFERENCE PIPELINE")
    print("=" * 80)

    # Load model if not provided
    if model is None:
        if model_path is None:
            model_dir = os.path.join(base_dir, 'code', 'business_entity_resolution', 'models')
            model_path = os.path.join(model_dir, 'lgbm_model_full.pkl')
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, 'lgbm_model.pkl')
        model, feature_cols, threshold = load_model(model_path)
        print(f"  Loaded model from {model_path}, threshold={threshold:.3f}")

    s1, s2, s3, _ = load_data(base_dir, 'test')

    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined S2+S3: {len(s23)}")
    del s2, s3
    gc.collect()

    all_s1_ids = set(s1['entity_id'].values)
    all_predictions = {}
    all_candidates = {}

    # Process in chunks for memory efficiency
    n_chunks = max(1, len(s1) // chunk_size)
    s1_chunks = np.array_split(s1, n_chunks)

    print(f"\n  Processing {len(s1)} S1 entities in {n_chunks} chunks of ~{chunk_size}")

    for i, s1_chunk in enumerate(s1_chunks):
        print(f"\n{'='*40}")
        print(f"  Chunk {i+1}/{n_chunks}: {len(s1_chunk)} S1 entities")
        print(f"{'='*40}")

        chunk_result = run_pipeline_chunk(
            s1_chunk, s23, gt_df=None,
            model=model, feature_cols=feature_cols, threshold=threshold,
            mode='infer', use_tfidf=True, n_neighbors=10
        )
        chunk_candidates, chunk_features, chunk_scores = chunk_result

        # Generate predictions for this chunk
        chunk_s1_ids = set(s1_chunk['entity_id'].values)
        chunk_preds = generate_predictions(chunk_features, chunk_scores, threshold, chunk_s1_ids)

        all_predictions.update(chunk_preds)
        for s1_id, cands in chunk_candidates.items():
            all_candidates[s1_id] = cands

        gc.collect()

    # Save outputs
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    save_predictions(all_predictions, matching_path)
    save_candidate_pairs(all_candidates, all_s1_ids, candidate_path)

    return all_predictions, all_candidates


def main():
    parser = argparse.ArgumentParser(description='Business Entity Resolution Pipeline')
    parser.add_argument('--mode', choices=['train', 'train-full', 'infer', 'full'],
                        default='full', help='Pipeline mode')
    parser.add_argument('--base-dir', default='/Users/daxsavaliya/Desktop/Aura_submission',
                        help='Base directory')
    parser.add_argument('--sample', type=float, default=None,
                        help='Sample fraction for training (e.g. 0.01 for 1%%)')
    parser.add_argument('--val-fraction', type=float, default=0.15,
                        help='Validation split fraction')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Chunk size for inference')
    args = parser.parse_args()

    if args.mode == 'train':
        model, feature_cols, threshold, f05 = run_train(
            args.base_dir,
            val_fraction=args.val_fraction,
            sample_fraction=args.sample
        )
        print(f"\n✓ Training complete. Val F_0.5: {f05:.4f}")

    elif args.mode == 'train-full':
        model, feature_cols, threshold = run_train_full(
            args.base_dir, sample_fraction=args.sample
        )
        print(f"\n✓ Full training complete. Threshold: {threshold:.3f}")

    elif args.mode == 'infer':
        run_inference(args.base_dir, chunk_size=args.chunk_size)
        print(f"\n✓ Inference complete.")

    elif args.mode == 'full':
        # Step 1: Train with validation
        model, feature_cols, threshold, f05 = run_train(
            args.base_dir,
            val_fraction=args.val_fraction,
            sample_fraction=args.sample
        )
        print(f"\n✓ Training complete. Val F_0.5: {f05:.4f}")

        # Step 2: Retrain on full data
        model, feature_cols, threshold = run_train_full(
            args.base_dir, sample_fraction=args.sample
        )

        # Step 3: Inference
        run_inference(
            args.base_dir,
            model=model, feature_cols=feature_cols, threshold=threshold,
            chunk_size=args.chunk_size
        )
        print(f"\n✓ Full pipeline complete.")


if __name__ == '__main__':
    main()
