#!/usr/bin/env python3
"""
BLOCKING OPTIMIZATION & SCALE BENCHMARK SCRIPT:
1. Candidate cap sweep (25, 50, 75, 100, 150, 200) on 10,000 entities across full 10.3M S2/S3 records.
2. Failure case analysis: Pull 30 missed cases and inspect rank / retrieval status.
3. Propose and test enhanced blocking:
   - IDF-weighted token ranking
   - Dual-pass name + address retrieval (always querying address tokens/digits)
   - Prefix/collapsed token indexing
4. Report the optimal candidate cap & configuration maximizing Macro F0.5.
"""

import os, sys, time, gc, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from normalize import normalize_name, normalize_address, extract_digits
from features import compute_pair_features, compute_features_batch
from model import load_model, predict_scores, compute_f05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "models", "lgbm_model.pkl")

# Load full data
print("Loading dataset...")
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

# Validation sample (5,000 US + 5,000 India)
np.random.seed(999)
us_val = s1_full[s1_full['country'] == 'US'].sample(n=5000, random_state=999)
india_val = s1_full[s1_full['country'] == 'India'].sample(n=5000, random_state=999)
val_s1 = pd.concat([us_val, india_val], ignore_index=True).reset_index(drop=True)
val_gt = gt_full[gt_full['source1_entity_id'].isin(val_s1['entity_id'])].reset_index(drop=True)

gt_map = {}
for _, row in val_gt.iterrows():
    sid = row['source1_entity_id']
    m = row['matched_entity_ids']
    if pd.isna(m) or str(m).strip() == '':
        gt_map[sid] = set()
    else:
        gt_map[sid] = set(str(m).split(','))

total_true_matches = sum(len(v) for v in gt_map.values())
print(f"Validation: {len(val_s1)} entities, {total_true_matches} true match pairs across {len(s23_full):,} S2/S3 records.")

# Pre-normalize S23
print("Normalizing S2/S3 records and building index...")
t0 = time.time()
s23_names = s23_full['business_name'].values
s23_addrs = s23_full['business_address'].values
s23_ids = s23_full['entity_id'].values
s23_countries = s23_full['country'].values

s23_names_norm = [normalize_name(x) for x in s23_names]
s23_addrs_norm = [normalize_address(x) for x in s23_addrs]

# Build country indices
name_idx = defaultdict(lambda: defaultdict(list))
addr_idx = defaultdict(lambda: defaultdict(list))
digit_idx = defaultdict(lambda: defaultdict(list))

for i in range(len(s23_full)):
    c = s23_countries[i]
    name_parts = s23_names_norm[i].split()
    for p in name_parts:
        if len(p) >= 3:
            name_idx[c][p].append(i)
    # 2-letter words or initials
    for p in name_parts:
        if len(p) == 2:
            name_idx[c][p].append(i)
    for t in s23_addrs_norm[i].split():
        if len(t) >= 4:
            addr_idx[c][t].append(i)
    for d in extract_digits(s23_addrs[i]):
        if len(d) >= 3:
            digit_idx[c][d].append(i)

print(f"Index built in {time.time()-t0:.1f}s")

s1_names = val_s1['business_name'].values
s1_addrs = val_s1['business_address'].values
s1_ids = val_s1['entity_id'].values
s1_countries = val_s1['country'].values
s1_names_norm = [normalize_name(x) for x in s1_names]
s1_addrs_norm = [normalize_address(x) for x in s1_addrs]

# Compute token document frequencies for IDF weighting
doc_freq_name = {c: {tok: len(lst) for tok, lst in name_idx[c].items()} for c in name_idx}
doc_freq_addr = {c: {tok: len(lst) for tok, lst in addr_idx[c].items()} for c in addr_idx}

# Enhanced multi-pass candidate generator with full rank scoring
def retrieve_and_rank_candidates(s1_idx, max_retrieve=500):
    sid = s1_ids[s1_idx]
    c = s1_countries[s1_idx]
    s1_n_norm = s1_names_norm[s1_idx]
    s1_a_norm = s1_addrs_norm[s1_idx]
    s1_n_parts = s1_n_norm.split()
    s1_a_parts = s1_a_norm.split()
    
    cand_indices = set()
    
    # 1. Name tokens with bucket size cap <= 3000
    for p in s1_n_parts:
        if p in name_idx[c] and len(name_idx[c][p]) <= 3000:
            cand_indices.update(name_idx[c][p])
            
    # 2. Always query address tokens with high selectivity
    for t in s1_a_parts:
        if len(t) >= 4 and t in addr_idx[c] and len(addr_idx[c][t]) <= 1000:
            cand_indices.update(addr_idx[c][t])
            
    # 3. Query digits (pincode/postal code or street number)
    for d in extract_digits(s1_addrs[s1_idx]):
        if len(d) >= 4 and d in digit_idx[c] and len(digit_idx[c][d]) <= 1000:
            cand_indices.update(digit_idx[c][d])
            
    if not cand_indices:
        return []
        
    # Rank candidates using IDF-weighted scoring
    s1_n_set = set(s1_n_parts)
    s1_a_set = set(s1_a_parts)
    
    scored = []
    N_c = len(s23_full)
    for ci in cand_indices:
        score = 0.0
        c_n_norm = s23_names_norm[ci]
        c_a_norm = s23_addrs_norm[ci]
        
        # Exact name match bonus
        if s1_n_norm == c_n_norm and s1_n_norm:
            score += 15.0
            
        c_n_set = set(c_n_norm.split())
        shared_name = s1_n_set & c_n_set
        if shared_name:
            # IDF weighted name score
            for tok in shared_name:
                df = doc_freq_name[c].get(tok, 1)
                idf = math.log((N_c + 1) / (df + 1))
                score += 2.0 * idf
                
        c_a_set = set(c_a_norm.split())
        shared_addr = s1_a_set & c_a_set
        if shared_addr:
            for tok in shared_addr:
                df = doc_freq_addr[c].get(tok, 1)
                idf = math.log((N_c + 1) / (df + 1))
                score += 1.0 * idf
                
        scored.append((score, ci))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ci for _, ci in scored[:max_retrieve]]

print("\nGenerating all retrieved candidates for 10,000 entities...")
t0 = time.time()
all_retrieved_cands = []
for i in range(len(val_s1)):
    c_list = retrieve_and_rank_candidates(i, max_retrieve=300)
    all_retrieved_cands.append(c_list)
print(f"Candidate retrieval completed in {time.time()-t0:.1f}s")

# Load trained model
model, feature_cols, base_thresh = load_model(MODEL_PATH)

print("\n" + "=" * 80)
print("1. CANDIDATE CAP SWEEP ON FULL 10.3M SCALE (Set B Benchmark)")
print("=" * 80)
print(f"{'Cap':<6} | {'Blocking Recall':<18} {'Avg Cands':<12} {'Max Cands':<10}")
print("-" * 65)

caps = [25, 50, 75, 100, 150, 200, 300]
cap_cands_dict = {}

for cap in caps:
    found_matches = 0
    total_cands = 0
    cands_map = {}
    for i in range(len(val_s1)):
        sid = s1_ids[i]
        top_cands = all_retrieved_cands[i][:cap]
        cand_ids = [s23_ids[ci] for ci in top_cands]
        cands_map[sid] = cand_ids
        total_cands += len(cand_ids)
        
        true_set = gt_map.get(sid, set())
        if true_set:
            found_matches += len(true_set & set(cand_ids))
            
    recall = found_matches / total_true_matches
    avg_cands = total_cands / len(val_s1)
    cap_cands_dict[cap] = cands_map
    print(f"{cap:<6} | {found_matches:>5}/{total_true_matches} ({recall*100:>5.2f}%)  {avg_cands:>10.1f}  {min(cap, avg_cands):>10.0f}")

# Evaluate Downstream Macro F0.5 at each cap for threshold 0.940 & 0.900
print("\n" + "=" * 80)
print("DOWNSTREAM MACRO F0.5 vs CANDIDATE CAP (At Threshold = 0.900 and 0.940)")
print("=" * 80)
print(f"{'Cap':<6} | {'Thresh 0.90: F0.5':<18} {'Micro Prec':<12} {'Micro Rec':<12} | {'Thresh 0.94: F0.5':<18} {'Micro Prec':<12} {'Micro Rec':<12}")
print("-" * 95)

def evaluate_predictions(scores, feats_df, s1_id_list, gt_dict, thresh):
    s1_arr = feats_df['s1_id'].values
    cand_arr = feats_df['cand_id'].values
    preds = defaultdict(set)
    for i in range(len(scores)):
        if scores[i] >= thresh:
            preds[s1_arr[i]].add(cand_arr[i])
            
    f05_list = []
    tp_total = fp_total = fn_total = 0
    for sid in s1_id_list:
        true_m = gt_dict.get(sid, set())
        pred_m = preds.get(sid, set())
        if not true_m:
            if not pred_m:
                f05_list.append(1.0)
            else:
                f05_list.append(0.0)
                fp_total += len(pred_m)
        else:
            tp = len(true_m & pred_m)
            fp = len(pred_m - true_m)
            fn = len(true_m - pred_m)
            tp_total += tp; fp_total += fp; fn_total += fn
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05_list.append(compute_f05(p, r))
            
    macro_f05 = np.mean(f05_list)
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    return macro_f05, micro_p, micro_r

for cap in [25, 50, 75, 100, 150, 200]:
    c_map = cap_cands_dict[cap]
    feats = compute_features_batch(val_s1, s23_full, c_map)
    scores = predict_scores(model, feats, feature_cols)
    
    f90, p90, r90 = evaluate_predictions(scores, feats, val_s1['entity_id'].tolist(), gt_map, thresh=0.900)
    f94, p94, r94 = evaluate_predictions(scores, feats, val_s1['entity_id'].tolist(), gt_map, thresh=0.940)
    print(f"{cap:<6} | {f90:<18.4f} {p90:<12.4f} {r90:<12.4f} | {f94:<18.4f} {p94:<12.4f} {r94:<12.4f}")

# SECTION 2: FAILURE CASE ANALYSIS (30 Cases not in Top-25)
print("\n" + "=" * 80)
print("2. DIAGNOSIS OF WHY RECALL CAPS AT 25: 30 SAMPLE MISSED CASES")
print("=" * 80)

# Build map from S23 ID to index
s23_id_to_idx = {s23_ids[i]: i for i in range(len(s23_ids))}

missed_cases = []
for i in range(len(val_s1)):
    sid = s1_ids[i]
    true_set = gt_map.get(sid, set())
    if not true_set:
        continue
    top25_set = set([s23_ids[ci] for ci in all_retrieved_cands[i][:25]])
    all_retrieved_set = set([s23_ids[ci] for ci in all_retrieved_cands[i]])
    
    for tid in true_set:
        if tid not in top25_set:
            # Find where it ranked
            rank = None
            if tid in all_retrieved_set:
                for r_idx, ci in enumerate(all_retrieved_cands[i]):
                    if s23_ids[ci] == tid:
                        rank = r_idx + 1
                        break
            
            t_idx = s23_id_to_idx.get(tid)
            t_name = s23_names[t_idx] if t_idx is not None else "N/A"
            t_addr = s23_addrs[t_idx] if t_idx is not None else "N/A"
            
            missed_cases.append({
                's1_id': sid, 's1_name': s1_names[i], 's1_addr': s1_addrs[i],
                'true_id': tid, 'true_name': t_name, 'true_addr': t_addr,
                'rank': rank if rank is not None else "Unretrieved (>300 or blocked)",
            })

print(f"Total true matches missed in Top-25: {len(missed_cases)}")
rank_counts = defaultdict(int)
for m in missed_cases:
    r = m['rank']
    if isinstance(r, int):
        if r <= 50:
            rank_counts['Rank 26-50'] += 1
        elif r <= 100:
            rank_counts['Rank 51-100'] += 1
        elif r <= 200:
            rank_counts['Rank 101-200'] += 1
        else:
            rank_counts['Rank 201-300'] += 1
    else:
        rank_counts['Not Retrieved (0 token overlap / filtered)'] += 1

print("\nRank Distribution of Missed Matches:")
for k, v in sorted(rank_counts.items()):
    print(f"  {k:<45}: {v:>5} ({v/len(missed_cases)*100:>5.1f}%)")

print("\n30 Sample Missed Cases:")
for idx, m in enumerate(missed_cases[:30]):
    print(f"[{idx+1:02d}] Rank: {m['rank']}")
    print(f"     S1:   '{m['s1_name']}' | '{m['s1_addr']}'")
    print(f"     True: '{m['true_name']}' | '{m['true_addr']}'")

