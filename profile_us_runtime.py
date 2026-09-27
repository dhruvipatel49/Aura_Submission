#!/usr/bin/env python3
"""
Profile US inference speed directly to verify active CPU time per batch.
"""
import os, sys, time, gc, math
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))
from normalize import normalize_name, normalize_address, extract_digits
from features import compute_pair_features
from model import load_model, predict_scores

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "models", "lgbm_model.pkl")

model, feature_cols, base_thresh = load_model(MODEL_PATH)

print("Profiling US batch time...")
t_load = time.time()
s2_chunks = []
for chunk in pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep='\t', chunksize=500000):
    c_sub = chunk[chunk['country'] == 'US']
    if len(c_sub) > 0:
        s2_chunks.append(c_sub)
s2 = pd.concat(s2_chunks, ignore_index=True)
del s2_chunks; gc.collect()

s3_chunks = []
for chunk in pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep='\t', chunksize=500000):
    c_sub = chunk[chunk['country'] == 'US']
    if len(c_sub) > 0:
        s3_chunks.append(c_sub)
s3 = pd.concat(s3_chunks, ignore_index=True)
del s3_chunks; gc.collect()

s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3; gc.collect()

s23['business_name'] = s23['business_name'].fillna('')
s23['business_address'] = s23['business_address'].fillna('')
s23['country'] = s23['country'].fillna('US')
n_s23 = len(s23)

s23_ids = s23['entity_id'].values
s23_names = s23['business_name'].values
s23_addrs = s23['business_address'].values
s23_countries = s23['country'].values
s23_names_norm = [normalize_name(x) for x in s23_names]
s23_addrs_norm = [normalize_address(x) for x in s23_addrs]

# Build index
name_idx = defaultdict(list)
addr_idx = defaultdict(list)
digit_idx = defaultdict(list)
for i in range(n_s23):
    parts = s23_names_norm[i].split()
    for p in parts:
        if len(p) >= 3:
            name_idx[p].append(i)
        elif len(p) == 2:
            name_idx[p].append(i)
    for t in s23_addrs_norm[i].split():
        if len(t) >= 4:
            addr_idx[t].append(i)
    for d in extract_digits(s23_addrs[i]):
        if len(d) >= 4:
            digit_idx[d].append(i)

doc_freq_name = {tok: len(lst) for tok, lst in name_idx.items()}
doc_freq_addr = {tok: len(lst) for tok, lst in addr_idx.items()}

# Sample 1 batch of 25,000 S1
s1_all = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep='\t')
s1_us = s1_all[s1_all['country'] == 'US'].iloc[:25000].reset_index(drop=True)
del s1_all; gc.collect()

s1_names = s1_us['business_name'].fillna('').values
s1_addrs = s1_us['business_address'].fillna('').values
s1_ids = s1_us['entity_id'].values
s1_names_norm = [normalize_name(x) for x in s1_names]
s1_addrs_norm = [normalize_address(x) for x in s1_addrs]

t_b0 = time.time()
batch_pairs = []
batch_pair_s1 = []
batch_pair_cand = []
batch_pair_name_sim = []

for i in range(len(s1_us)):
    sid = s1_ids[i]
    s1_name_n = s1_names_norm[i]
    s1_addr_n = s1_addrs_norm[i]
    s1_n_parts = s1_name_n.split()
    s1_a_parts = s1_addr_n.split()
    cand_indices = set()
    for tok in s1_n_parts:
        if tok in name_idx and len(name_idx[tok]) <= 3000:
            cand_indices.update(name_idx[tok])
    for tok in s1_a_parts:
        if len(tok) >= 4 and tok in addr_idx and len(addr_idx[tok]) <= 1000:
            cand_indices.update(addr_idx[tok])
    for d in extract_digits(s1_addrs[i]):
        if len(d) >= 4 and d in digit_idx and len(digit_idx[d]) <= 1000:
            cand_indices.update(digit_idx[d])
            
    if len(cand_indices) > 25:
        s1_n_set = set(s1_n_parts)
        s1_a_set = set(s1_a_parts)
        scored = []
        for ci in cand_indices:
            c_name_n = s23_names_norm[ci]
            c_addr_n = s23_addrs_norm[ci]
            score = 0.0
            if s1_name_n == c_name_n and s1_name_n:
                score += 15.0
            c_n_set = set(c_name_n.split())
            shared_name = s1_n_set & c_n_set
            if shared_name:
                for tok in shared_name:
                    df = doc_freq_name.get(tok, 1)
                    score += 2.0 * math.log((n_s23 + 1) / (df + 1))
            c_a_set = set(c_addr_n.split())
            shared_addr = s1_a_set & c_a_set
            if shared_addr:
                for tok in shared_addr:
                    df = doc_freq_addr.get(tok, 1)
                    score += 1.0 * math.log((n_s23 + 1) / (df + 1))
            scored.append((score, ci))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected_cands = [ci for _, ci in scored[:25]]
    else:
        selected_cands = list(cand_indices)
        
    for ci in selected_cands:
        f = compute_pair_features(
            s1_names[i], s1_addrs[i], 'US', sid,
            s23_names[ci], s23_addrs[ci], 'US', s23_ids[ci],
            s1_name_n, s1_addr_n,
            s23_names_norm[ci], s23_addrs_norm[ci]
        )
        batch_pairs.append(f)
        batch_pair_s1.append(sid)
        batch_pair_cand.append(s23_ids[ci])

df_batch = pd.DataFrame(batch_pairs)
scores = predict_scores(model, df_batch, feature_cols)
elapsed = time.time() - t_b0
print(f"\n1 Batch (25,000 S1 entities, {len(batch_pairs)} pairs) completed in: {elapsed:.2f} seconds.")
print(f"Projected total US active runtime (27 batches): {27 * elapsed / 60:.2f} minutes.")
