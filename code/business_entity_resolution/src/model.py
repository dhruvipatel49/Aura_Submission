#!/usr/bin/env python3
"""
Matching model training, thresholding, and inference for Business Entity Resolution.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import lightgbm as lgb
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingClassifier
    print("WARNING: LightGBM not available, falling back to sklearn GradientBoostingClassifier")


def prepare_training_data(features_df, ground_truth_df):
    """
    Label feature vectors using ground truth.
    Vectorized: builds a set of (s1_id, cand_id) true-match pairs, then checks membership.
    """
    # Build ground truth lookup: set of (s1_id, cand_id) pairs
    true_pairs = set()
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids']
        if pd.notna(matched) and str(matched).strip():
            for cid in str(matched).split(','):
                true_pairs.add((s1_id, cid))

    # Vectorized labeling
    features_df = features_df.copy()
    s1_ids = features_df['s1_id'].values
    cand_ids = features_df['cand_id'].values
    labels = np.array([1 if (s1_ids[i], cand_ids[i]) in true_pairs else 0
                       for i in range(len(s1_ids))], dtype=np.int32)
    features_df['label'] = labels

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos > 0:
        print(f"  Training data: {n_pos} positive, {n_neg} negative (ratio 1:{n_neg/n_pos:.1f})")
    else:
        print(f"  Training data: 0 positive, {n_neg} negative")

    return features_df


def get_feature_cols(df):
    """Get feature column names (exclude IDs and label)."""
    exclude = {'s1_id', 'cand_id', 'label'}
    return [c for c in df.columns if c not in exclude]


def train_model(features_df, feature_cols=None):
    """
    Train a binary classifier (LightGBM preferred, sklearn fallback).

    Returns:
        (model, feature_cols)
    """
    if feature_cols is None:
        feature_cols = get_feature_cols(features_df)

    X = features_df[feature_cols].values
    y = features_df['label'].values

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"  Training on {len(X)} samples, {len(feature_cols)} features")
    print(f"  Positive: {n_pos}, Negative: {n_neg}, scale_pos_weight: {scale_pos_weight:.2f}")

    if HAS_LGBM:
        print("  Using LightGBM")
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'n_jobs': -1,
            'min_child_samples': 50,
            'max_depth': 8,
        }

        train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(100)],
        )

        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
        print("\n  Top 15 features by gain:")
        for fname, fimp in feat_imp[:15]:
            print(f"    {fname}: {fimp:.1f}")
    else:
        print("  Using sklearn GradientBoostingClassifier (fallback)")
        # Subsample for speed if large
        max_train = 500000
        if len(X) > max_train:
            idx = np.random.choice(len(X), max_train, replace=False)
            X_train, y_train = X[idx], y[idx]
            print(f"  Subsampled to {max_train} for sklearn speed")
        else:
            X_train, y_train = X, y

        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Feature importance
        importance = model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
        print("\n  Top 15 features by importance:")
        for fname, fimp in feat_imp[:15]:
            print(f"    {fname}: {fimp:.4f}")

    return model, feature_cols


def predict_scores(model, features_df, feature_cols):
    """
    Predict match probability for each pair.
    """
    X = features_df[feature_cols].values
    if HAS_LGBM and hasattr(model, 'predict'):
        scores = model.predict(X)
    else:
        scores = model.predict_proba(X)[:, 1]
    return scores


def compute_f05(precision, recall):
    """Compute F_0.5 score."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def optimize_threshold(features_df, scores, ground_truth_df, thresholds=None):
    """
    Find the threshold that maximizes macro-averaged F_0.5 on the dataset.
    Pre-groups data by s1_id for efficiency.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.02)

    # Build ground truth lookup
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids']
        if pd.isna(matched) or str(matched).strip() == '':
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = set(str(matched).split(','))

    # Pre-group: s1_id -> list of (cand_id, score)
    entity_data = {}
    s1_ids_arr = features_df['s1_id'].values
    cand_ids_arr = features_df['cand_id'].values
    for i in range(len(s1_ids_arr)):
        s1_id = s1_ids_arr[i]
        if s1_id not in entity_data:
            entity_data[s1_id] = []
        entity_data[s1_id].append((cand_ids_arr[i], scores[i]))

    all_s1_ids = set(gt_map.keys())

    best_threshold = 0.5
    best_f05 = 0.0

    for threshold in thresholds:
        entity_f05_scores = []

        for s1_id in all_s1_ids:
            true_matches = gt_map.get(s1_id, set())
            pairs = entity_data.get(s1_id, [])
            predicted = {cid for cid, sc in pairs if sc >= threshold}

            if not true_matches and not predicted:
                entity_f05_scores.append(1.0)
            elif not true_matches and predicted:
                entity_f05_scores.append(0.0)
            elif true_matches and not predicted:
                entity_f05_scores.append(0.0)
            else:
                tp = len(true_matches & predicted)
                fp = len(predicted - true_matches)
                fn = len(true_matches - predicted)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                entity_f05_scores.append(compute_f05(precision, recall))

        macro_f05 = np.mean(entity_f05_scores)
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = threshold

    print(f"\n  Best threshold: {best_threshold:.3f}, macro F_0.5: {best_f05:.4f}")

    # Compute detailed stats at best threshold
    tp_total = fp_total = fn_total = 0
    singleton_correct = singleton_wrong = 0

    for s1_id in all_s1_ids:
        true_matches = gt_map.get(s1_id, set())
        pairs = entity_data.get(s1_id, [])
        predicted = {cid for cid, sc in pairs if sc >= best_threshold}

        if not true_matches:
            if not predicted:
                singleton_correct += 1
            else:
                singleton_wrong += 1
                fp_total += len(predicted)
        else:
            tp = len(true_matches & predicted)
            fp = len(predicted - true_matches)
            fn = len(true_matches - predicted)
            tp_total += tp
            fp_total += fp
            fn_total += fn

    print(f"  At threshold {best_threshold:.3f}:")
    print(f"    TP={tp_total}, FP={fp_total}, FN={fn_total}")
    print(f"    Singletons correct: {singleton_correct}, wrong: {singleton_wrong}")
    if tp_total + fp_total > 0:
        print(f"    Micro precision: {tp_total/(tp_total+fp_total):.4f}")
    if tp_total + fn_total > 0:
        print(f"    Micro recall: {tp_total/(tp_total+fn_total):.4f}")

    return best_threshold, best_f05


def generate_predictions(features_df, scores, threshold, s1_ids_all):
    """
    Generate final matching results.

    Args:
        features_df: DataFrame with s1_id, cand_id columns
        scores: predicted probabilities
        threshold: classification threshold
        s1_ids_all: all S1 entity IDs that must appear in output

    Returns:
        dict: s1_id -> list of matched cand_ids
    """
    df = features_df.copy()
    df['score'] = scores

    predictions = {}

    # For entities with candidates
    for s1_id, group in df.groupby('s1_id'):
        matches = group[group['score'] >= threshold]['cand_id'].tolist()
        predictions[s1_id] = matches

    # Ensure all S1 entities appear (including those with no candidates)
    for s1_id in s1_ids_all:
        if s1_id not in predictions:
            predictions[s1_id] = []

    return predictions


def save_predictions(predictions, output_path):
    """Save predictions in the required TSV format."""
    rows = []
    for s1_id in sorted(predictions.keys()):
        matched = ','.join(predictions[s1_id])
        rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': matched})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved {len(df)} rows to {output_path}")
    return df


def save_candidate_pairs(candidates, s1_ids_all, output_path):
    """Save candidate pairs in the required TSV format."""
    rows = []
    for s1_id in sorted(s1_ids_all):
        cand_set = candidates.get(s1_id, set())
        cand_str = ','.join(sorted(cand_set))
        rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved {len(df)} rows to {output_path}")
    return df


def save_model(model, feature_cols, threshold, path):
    """Save model and metadata."""
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'threshold': threshold,
        }, f)
    print(f"  Model saved to {path}")


def load_model(path):
    """Load model and metadata."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['feature_cols'], data['threshold']
