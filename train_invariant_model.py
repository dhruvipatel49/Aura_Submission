#!/usr/bin/env python3
"""
Retrain LightGBM with improved blocking (prefix+bigram indexes) and
new features (jaro_winkler, postal_exact, name_digits).
Threshold is optimized on a HELD-OUT validation split, not training data.
"""
import os, sys, time, gc, pickle, math
import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from normalize import normalize_name, normalize_address, extract_digits
from features import compute_features_batch
from model import prepare_training_data, optimize_threshold, save_model, compute_f05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading training data...")
s1_train = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep='\t')
s2_train = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep='\t')
s3_train = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep='\t')
gt_train = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep='\t')

s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
s23_train['business_name'] = s23_train['business_name'].fillna('')
s23_train['business_address'] = s23_train['business_address'].fillna('')
s1_train['business_name'] = s1_train['business_name'].fillna('')
s1_train['business_address'] = s1_train['business_address'].fillna('')
del s2_train, s3_train; gc.collect()

# ---- Build improved blocking (same as new run_full.py) ----
def get_tokens(name_norm):
    toks = set()
    parts = name_norm.split()
    for p in parts:
        if len(p) >= 3:
            toks.add(p)
    collapsed = name_norm.replace(' ', '')
    if len(collapsed) >= 5 and len(parts) > 1:
        toks.add(collapsed)
    return toks

def get_name_prefixes(name_norm, lengths=(4, 5, 6)):
    prefixes = set()
    parts = name_norm.split()
    for p in parts:
        for l in lengths:
            if len(p) >= l:
                prefixes.add(p[:l])
    collapsed = name_norm.replace(' ', '')
    for l in lengths:
        if len(collapsed) >= l:
            prefixes.add('_c_' + collapsed[:l])
    return prefixes

def get_name_bigrams(name_norm, n=4):
    collapsed = name_norm.replace(' ', '')
    if len(collapsed) < n:
        return set()
    return {collapsed[i:i+n] for i in range(len(collapsed) - n + 1)}

def get_cands_improved(s1_df, s23_df, max_cands=50):
    s23_names = s23_df['business_name'].values
    s23_addrs = s23_df['business_address'].values
    s23_ids = s23_df['entity_id'].values
    s23_countries = s23_df['country'].values
    s23_names_norm = [normalize_name(x) for x in s23_names]
    s23_addrs_norm = [normalize_address(x) for x in s23_addrs]

    n_s23 = len(s23_df)

    name_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    digit_idx = defaultdict(list)
    prefix_idx = defaultdict(list)
    bigram_idx = defaultdict(list)

    for i in range(n_s23):
        for tok in get_tokens(s23_names_norm[i]):
            name_idx[tok].append(i)
        for tok in s23_names_norm[i].split():
            if len(tok) == 2:
                name_idx[tok].append(i)
        for pfx in get_name_prefixes(s23_names_norm[i]):
            prefix_idx[pfx].append(i)
        for bg in get_name_bigrams(s23_names_norm[i]):
            bigram_idx[bg].append(i)
        for tok in s23_addrs_norm[i].split():
            if len(tok) >= 4:
                addr_idx[tok].append(i)
        for d in extract_digits(s23_addrs[i]):
            if len(d) >= 3:
                digit_idx[d].append(i)

    doc_freq_name = {tok: len(lst) for tok, lst in name_idx.items()}
    doc_freq_addr = {tok: len(lst) for tok, lst in addr_idx.items()}

    s1_names_norm = [normalize_name(x) for x in s1_df['business_name'].values]
    s1_addrs_norm = [normalize_address(x) for x in s1_df['business_address'].values]
    s1_addrs_raw = s1_df['business_address'].values
    s1_ids = s1_df['entity_id'].values

    MAX_NAME_BUCKET = 5000
    MAX_ADDR_BUCKET = 2000
    MAX_PREFIX_BUCKET = 4000
    MAX_BIGRAM_BUCKET = 3000
    MIN_SHARED_BIGRAMS = 2

    cands_dict = {}
    for i in range(len(s1_df)):
        sid = s1_ids[i]
        s1_name_n = s1_names_norm[i]
        s1_addr_n = s1_addrs_norm[i]

        cands = set()

        # 1. Name token index
        for p in get_tokens(s1_name_n):
            if p in name_idx and len(name_idx[p]) <= MAX_NAME_BUCKET:
                cands.update(name_idx[p])

        # 2. Prefix index
        for pfx in get_name_prefixes(s1_name_n):
            if pfx in prefix_idx and len(prefix_idx[pfx]) <= MAX_PREFIX_BUCKET:
                cands.update(prefix_idx[pfx])

        # 3. Bigram index
        s1_bigrams = get_name_bigrams(s1_name_n)
        if s1_bigrams:
            bigram_hits = defaultdict(int)
            for bg in s1_bigrams:
                bucket = bigram_idx.get(bg)
                if bucket and len(bucket) <= MAX_BIGRAM_BUCKET:
                    for ci in bucket:
                        bigram_hits[ci] += 1
            threshold_bg = max(MIN_SHARED_BIGRAMS, len(s1_bigrams) // 3)
            for ci, cnt in bigram_hits.items():
                if cnt >= threshold_bg:
                    cands.add(ci)

        # 4. Address token index
        for t in s1_addr_n.split():
            if len(t) >= 4 and t in addr_idx and len(addr_idx[t]) <= MAX_ADDR_BUCKET:
                cands.update(addr_idx[t])

        # 5. Digit index (fixed to >=3)
        for d in extract_digits(str(s1_addrs_raw[i])):
            if len(d) >= 3 and d in digit_idx and len(digit_idx[d]) <= MAX_ADDR_BUCKET:
                cands.update(digit_idx[d])

        # Rank and cap
        if len(cands) > max_cands:
            s1_n_toks = set(s1_name_n.split())
            s1_a_toks = set(s1_addr_n.split())
            scored = []
            for ci in cands:
                sc = 0.0
                if s1_name_n == s23_names_norm[ci] and s1_name_n:
                    sc += 15.0
                c_n_toks = set(s23_names_norm[ci].split())
                if s1_n_toks and c_n_toks:
                    shared = s1_n_toks & c_n_toks
                    for tok in shared:
                        df = doc_freq_name.get(tok, 1)
                        sc += 2.0 * math.log((n_s23 + 1) / (df + 1))
                c_a_toks = set(s23_addrs_norm[ci].split())
                if s1_a_toks and c_a_toks:
                    shared_a = s1_a_toks & c_a_toks
                    for tok in shared_a:
                        df = doc_freq_addr.get(tok, 1)
                        sc += 1.0 * math.log((n_s23 + 1) / (df + 1))
                scored.append((sc, ci))
            scored.sort(key=lambda x: x[0], reverse=True)
            cands = [ci for _, ci in scored[:max_cands]]

        cands_dict[sid] = [s23_ids[ci] for ci in cands]
    return cands_dict


# ---- Sample: 20k US + 20k India for training (increased from 15k+15k) ----
np.random.seed(42)
us_s1 = s1_train[s1_train['country'] == 'US'].sample(n=20000, random_state=42)
india_s1 = s1_train[s1_train['country'] == 'India'].sample(n=20000, random_state=42)

# Hold out 20% for threshold optimization (not used in training)
us_train = us_s1.iloc[:16000]
us_val = us_s1.iloc[16000:]
india_train = india_s1.iloc[:16000]
india_val = india_s1.iloc[16000:]

s1_train_split = pd.concat([us_train, india_train], ignore_index=True)
s1_val_split = pd.concat([us_val, india_val], ignore_index=True)

print(f"Train: {len(s1_train_split)}, Val: {len(s1_val_split)}")

# Generate candidates with improved blocking for training split
print("Generating training candidates with improved blocking...")
t0 = time.time()
cands_train = get_cands_improved(s1_train_split, s23_train)
print(f"Training candidates done in {time.time()-t0:.1f}s")

print("Generating validation candidates with improved blocking...")
t0 = time.time()
cands_val = get_cands_improved(s1_val_split, s23_train)
print(f"Val candidates done in {time.time()-t0:.1f}s")

print("Extracting training features...")
t0 = time.time()
feats_train_df = compute_features_batch(s1_train_split, s23_train, cands_train)
labeled_train_df = prepare_training_data(feats_train_df, gt_train)
print(f"Training features done in {time.time()-t0:.1f}s, {len(feats_train_df)} pairs")

print("Extracting validation features...")
t0 = time.time()
feats_val_df = compute_features_batch(s1_val_split, s23_train, cands_val)
labeled_val_df = prepare_training_data(feats_val_df, gt_train)
print(f"Validation features done in {time.time()-t0:.1f}s, {len(feats_val_df)} pairs")

# ---- Feature columns: country-invariant only ----
country_specific_features = {
    'name_len_s1', 'name_len_cand', 'name_len_diff',
    'name_token_count_s1', 'name_token_count_cand', 'name_token_count_diff',
    'addr_len_diff', 'addr_digit_overlap',
    's1_name_is_non_latin', 'cand_name_is_non_latin'
}
feature_cols = [c for c in feats_train_df.columns
                if c not in {'s1_id', 'cand_id', 'label'} and c not in country_specific_features]
print(f"Using {len(feature_cols)} country-invariant features: {feature_cols}")

X_train = labeled_train_df[feature_cols].fillna(0).values
y_train = labeled_train_df['label'].values
X_val = labeled_val_df[feature_cols].fillna(0).values
y_val = labeled_val_df['label'].values

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"Train: {int(y_train.sum())} pos, {int((y_train==0).sum())} neg, SPW={scale_pos_weight:.1f}")
print(f"Val:   {int(y_val.sum())} pos, {int((y_val==0).sum())} neg")

# ---- Train LightGBM ----
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 127,        # Increased from 63
    'learning_rate': 0.03,    # Smaller LR with more rounds
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'n_jobs': -1,
    'min_child_samples': 30,  # Reduced from 50 to allow finer splits
    'max_depth': 10,          # Increased from 8
    'lambda_l1': 0.1,         # Regularization
    'lambda_l2': 0.1,
}

print(f"Training LightGBM on {len(X_train)} pairs...")
t0 = time.time()
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=800,      # More rounds (early stopping handles it)
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.log_evaluation(100),
        lgb.early_stopping(50, verbose=True),  # Early stopping on validation
    ],
)
print(f"Training done in {time.time()-t0:.1f}s")

# Feature importance
importance = model.feature_importance(importance_type='gain')
feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
print("\nTop 20 features by gain:")
for fname, fimp in feat_imp[:20]:
    print(f"  {fname}: {fimp:.1f}")

# ---- Optimize threshold on VALIDATION data (not training!) ----
print("\nOptimizing threshold on HELD-OUT validation split...")
val_scores = model.predict(X_val)
best_thresh, best_f05 = optimize_threshold(labeled_val_df, val_scores, gt_train)
print(f"Best threshold: {best_thresh:.3f}, Val macro F0.5: {best_f05:.4f}")

# Save model
model_path = os.path.join(MODEL_DIR, "lgbm_model.pkl")
save_model(model, feature_cols, best_thresh, model_path)
print(f"\nModel saved: {model_path}")
print(f"Threshold: {best_thresh:.3f}, Val Macro F0.5: {best_f05:.4f}")
print(f"Feature count: {len(feature_cols)}")
