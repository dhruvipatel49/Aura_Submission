#!/usr/bin/env python3
"""
Full-scale inference pipeline for Business Entity Resolution.
Processes test data country-by-country (France, US, India) using
vectorized token inverted indexing, fast pair feature computation,
and LightGBM prediction with F0.5-optimized thresholding.

Outputs:
  output/matching_results.tsv
  output/candidate_pairs.tsv
"""

import os
import sys
import time
import gc
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'code', 'business_entity_resolution', 'src'))

from normalize import normalize_name, normalize_address, extract_digits
from features import compute_pair_features
from model import load_model, predict_scores

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "models", "lgbm_model.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_tokens(name_norm):
    """Extract tokens and collapsed form for index matching."""
    toks = set()
    parts = name_norm.split()
    for p in parts:
        if len(p) >= 3:
            toks.add(p)
    # Add collapsed version (e.g., 'starbuckscoffee' or 'georgesaul')
    collapsed = name_norm.replace(' ', '')
    if len(collapsed) >= 5 and len(parts) > 1:
        toks.add(collapsed)
    return toks


def get_name_prefixes(name_norm, lengths=(4, 5, 6)):
    """Extract fixed-length prefixes from each token and from the collapsed name.
    Used as a lightweight fuzzy-recall index to catch typos and abbreviations."""
    prefixes = set()
    parts = name_norm.split()
    for p in parts:
        for l in lengths:
            if len(p) >= l:
                prefixes.add(p[:l])
    # Also prefix of the full concatenated name
    collapsed = name_norm.replace(' ', '')
    for l in lengths:
        if len(collapsed) >= l:
            prefixes.add('_c_' + collapsed[:l])  # namespace to avoid collision with token prefixes
    return prefixes


def get_name_bigrams(name_norm, n=3):
    """Extract character n-grams from the collapsed normalized name.
    Catches transliteration variants and single-character typos."""
    collapsed = name_norm.replace(' ', '')
    if len(collapsed) < n:
        return set()
    return {collapsed[i:i+n] for i in range(len(collapsed) - n + 1)}


def get_addr_tokens(addr_norm):
    """Extract significant address tokens (>= 4 chars, skipping generic words)."""
    skip = {
        'street', 'road', 'avenue', 'lane', 'drive', 'floor', 'block',
        'near', 'post', 'delhi', 'mumbai', 'california', 'texas', 'india',
        'unit', 'building', 'nagar', 'colony', 'pradesh', 'house', 'plot',
        'first', 'second', 'third', 'state', 'city', 'north', 'south', 'east', 'west'
    }
    return {t for t in addr_norm.split() if len(t) >= 4 and t not in skip}


import math

COUNTRY_THRESHOLDS = {
    "France": 0.960,  # Conservative threshold for unseen country
    "US": 0.880,      # Tuned on holdout (improved blocking increases precision headroom)
    "India": 0.860,   # India has more name variation; slightly lower threshold
}


def process_country(country, model, feature_cols,
                    out_matches_f, out_cands_f):
    """
    Process one country end-to-end:
    Loads S2+S3, builds multi-pass inverted index, processes S1 in chunks,
    applies IDF-weighted candidate ranking, LightGBM scoring with country-specific
    calibrated thresholding and co-location guards.
    """
    threshold = COUNTRY_THRESHOLDS.get(country, 0.950)
    print(f"\n{'='*70}")
    print(f"PROCESSING COUNTRY: {country} (Threshold: {threshold:.3f})")
    print(f"{'='*70}")
    t_start = time.time()

    # Step 1: Load S2 and S3 for this country
    print(f"[{country}] Loading S2 and S3...")
    t0 = time.time()
    s2_chunks = []
    for chunk in pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep='\t', chunksize=500000):
        c_sub = chunk[chunk['country'] == country]
        if len(c_sub) > 0:
            s2_chunks.append(c_sub)
    s2 = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])
    del s2_chunks; gc.collect()

    s3_chunks = []
    for chunk in pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep='\t', chunksize=500000):
        c_sub = chunk[chunk['country'] == country]
        if len(c_sub) > 0:
            s3_chunks.append(c_sub)
    s3 = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])
    del s3_chunks; gc.collect()

    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3; gc.collect()

    s23['business_name'] = s23['business_name'].fillna('')
    s23['business_address'] = s23['business_address'].fillna('')
    s23['country'] = s23['country'].fillna(country)

    n_s23 = len(s23)
    print(f"[{country}] S2+S3 loaded: {n_s23} records in {time.time()-t0:.1f}s")

    # Step 2: Pre-normalize S23
    t0 = time.time()
    s23_ids = s23['entity_id'].values
    s23_names = s23['business_name'].values
    s23_addrs = s23['business_address'].values
    s23_countries = s23['country'].values

    s23_names_norm = [normalize_name(x) for x in s23_names]
    s23_addrs_norm = [normalize_address(x) for x in s23_addrs]
    print(f"[{country}] S2+S3 normalized in {time.time()-t0:.1f}s")

    # Step 3: Build multi-pass inverted indexes
    t0 = time.time()
    name_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    digit_idx = defaultdict(list)
    prefix_idx = defaultdict(list)   # NEW: prefix-based for typo robustness
    bigram_idx = defaultdict(list)   # NEW: char-4gram for transliteration/typo recall

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
        for tok in get_addr_tokens(s23_addrs_norm[i]):
            addr_idx[tok].append(i)
        for d in extract_digits(s23_addrs[i]):
            if len(d) >= 3:
                digit_idx[d].append(i)
                if len(d) > 3:
                    digit_idx[d[:3]].append(i)

    doc_freq_name = {tok: len(lst) for tok, lst in name_idx.items()}
    doc_freq_addr = {tok: len(lst) for tok, lst in addr_idx.items()}

    print(f"[{country}] Inverted indexes built: {len(name_idx)} name, {len(prefix_idx)} prefix, {len(bigram_idx)} bigram, {len(addr_idx)} addr, {len(digit_idx)} digit tokens in {time.time()-t0:.1f}s")

    # Step 4: Load S1 for this country
    t0 = time.time()
    s1_all = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep='\t')
    s1 = s1_all[s1_all['country'] == country].copy().reset_index(drop=True)
    del s1_all; gc.collect()

    s1['business_name'] = s1['business_name'].fillna('')
    s1['business_address'] = s1['business_address'].fillna('')
    s1['country'] = s1['country'].fillna(country)

    n_s1 = len(s1)
    print(f"[{country}] S1 loaded: {n_s1} records in {time.time()-t0:.1f}s")

    s1_ids = s1['entity_id'].values
    s1_names = s1['business_name'].values
    s1_addrs = s1['business_address'].values
    s1_countries = s1['country'].values
    s1_names_norm = [normalize_name(x) for x in s1_names]
    s1_addrs_norm = [normalize_address(x) for x in s1_addrs]

    # Step 5: Process S1 in batches
    BATCH_SIZE = 25000
    n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE

    total_matches = 0
    total_singletons = 0
    total_candidates = 0

    MAX_CANDS_PER_ENTITY = 50    # Increased from 25: reduces ranking-cap recall loss
    MAX_NAME_BUCKET = 5000       # Increased from 3000: fewer silently-skipped tokens
    MAX_ADDR_BUCKET = 2000       # Increased from 1000
    MAX_PREFIX_BUCKET = 4000     # New: prefix index bucket cap
    MAX_BIGRAM_BUCKET = 3000     # New: bigram index bucket cap
    MIN_SHARED_BIGRAMS = 2       # New: minimum shared 4-grams to add from bigram index

    print(f"[{country}] Processing {n_s1} S1 entities across {n_batches} batches...")
    for b in range(n_batches):
        b_start = b * BATCH_SIZE
        b_end = min(n_s1, (b + 1) * BATCH_SIZE)
        t_b0 = time.time()

        batch_pairs = []
        batch_pair_s1 = []
        batch_pair_cand = []
        batch_pair_name_sim = []
        batch_candidates_map = {}

        for i in range(b_start, b_end):
            sid = s1_ids[i]
            s1_name_n = s1_names_norm[i]
            s1_addr_n = s1_addrs_norm[i]
            s1_c = s1_countries[i]

            s1_name_toks = get_tokens(s1_name_n)
            s1_n_parts = s1_name_n.split()
            s1_a_parts = s1_addr_n.split()
            cand_indices = set()

            # 1. Retrieve by name tokens
            for tok in s1_name_toks:
                bucket = name_idx.get(tok)
                if bucket and len(bucket) <= MAX_NAME_BUCKET:
                    cand_indices.update(bucket)

            # 2. Prefix-based lookup (catches abbreviations/typos with shared prefix)
            for pfx in get_name_prefixes(s1_name_n):
                bucket = prefix_idx.get(pfx)
                if bucket and len(bucket) <= MAX_PREFIX_BUCKET:
                    cand_indices.update(bucket)

            # 3. Character 4-gram lookup (catches typos, transliteration variants)
            s1_bigrams = get_name_bigrams(s1_name_n)
            if s1_bigrams:
                bigram_hits = defaultdict(int)
                for bg in s1_bigrams:
                    bucket = bigram_idx.get(bg)
                    if bucket and len(bucket) <= MAX_BIGRAM_BUCKET:
                        for ci in bucket:
                            bigram_hits[ci] += 1
                # Only add candidates sharing >= MIN_SHARED_BIGRAMS (avoids noise)
                threshold_bg = max(MIN_SHARED_BIGRAMS, len(s1_bigrams) // 3)
                for ci, cnt in bigram_hits.items():
                    if cnt >= threshold_bg:
                        cand_indices.add(ci)

            # 4. Always query selective address tokens
            for tok in s1_a_parts:
                if len(tok) >= 4:
                    bucket = addr_idx.get(tok)
                    if bucket and len(bucket) <= MAX_ADDR_BUCKET:
                        cand_indices.update(bucket)

            # 5. Query selective address digits (FIXED: >=3 to match index, was >=4)
            for d in extract_digits(s1_addrs[i]):
                if len(d) >= 3:
                    bucket = digit_idx.get(d)
                    if bucket and len(bucket) <= MAX_ADDR_BUCKET:
                        cand_indices.update(bucket)

            # 6. ADDRESS-ONLY & DESPERATE FALLBACK
            if not cand_indices:
                # 6a. Try digits >= 3 chars first with relaxed cap
                for d in extract_digits(s1_addrs[i]):
                    if len(d) >= 3:
                        bucket = digit_idx.get(d)
                        if bucket and len(bucket) <= 10000:
                            cand_indices.update(bucket)
                        if len(d) > 3:
                            bucket_pfx = digit_idx.get(d[:3])
                            if bucket_pfx and len(bucket_pfx) <= 10000:
                                cand_indices.update(bucket_pfx)
                
                # 6b. Try address tokens with tight cap
                if not cand_indices:
                    for tok in s1_a_parts:
                        if len(tok) >= 4:
                            bucket = addr_idx.get(tok)
                            if bucket and len(bucket) <= 1000:
                                cand_indices.update(bucket)
                
                # 6c. DESPERATE PASS: If still empty, the name might be common (like 'urology').
                # Pull large name buckets (up to 30000) and intersect with ANY address/digit token.
                if not cand_indices:
                    desperate_cands = set()
                    for tok in s1_name_toks:
                        bucket = name_idx.get(tok)
                        if bucket and 5000 < len(bucket) <= 30000:
                            desperate_cands.update(bucket)
                    
                    if desperate_cands:
                        # Find all candidates that have SOME digit or address token matching
                        valid_cands = set()
                        for d in extract_digits(s1_addrs[i]):
                            if len(d) >= 3:
                                bucket = digit_idx.get(d)
                                if bucket: valid_cands.update(bucket)
                                if len(d) > 3:
                                    bucket = digit_idx.get(d[:3])
                                    if bucket: valid_cands.update(bucket)
                        for tok in s1_a_parts:
                            if len(tok) >= 4:
                                bucket = addr_idx.get(tok)
                                if bucket: valid_cands.update(bucket)
                        
                        cand_indices = desperate_cands & valid_cands

            # If candidate set is large, prioritize using IDF-weighted similarity:
            if len(cand_indices) > MAX_CANDS_PER_ENTITY:
                # Pre-compute 4-gram set for s1 name (for fuzzy ranking bonus)
                s1_4grams = get_name_bigrams(s1_name_n)

                scored_cands = []
                s1_n_set = set(s1_n_parts)
                s1_a_set = set(s1_a_parts)

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
                            idf = math.log((n_s23 + 1) / (df + 1))
                            score += 2.0 * idf

                    c_a_set = set(c_addr_n.split())
                    shared_addr = s1_a_set & c_a_set
                    if shared_addr:
                        for tok in shared_addr:
                            df = doc_freq_addr.get(tok, 1)
                            idf = math.log((n_s23 + 1) / (df + 1))
                            score += 1.0 * idf

                    # FIX: Add 4-gram Jaccard bonus so fuzzy candidates (from prefix/bigram
                    # indexes) don't score 0.0 and get discarded before the classifier sees them.
                    # This is additive — it doesn't replace the exact-token IDF signal above.
                    if s1_4grams:
                        c_4grams = get_name_bigrams(c_name_n)
                        if c_4grams:
                            union_size = len(s1_4grams | c_4grams)
                            if union_size > 0:
                                jaccard_4g = len(s1_4grams & c_4grams) / union_size
                                # Weight: 5.0 * jaccard keeps fuzzy matches competitive
                                # but below a 2-token exact IDF match (~4.0+). This means
                                # pure fuzzy candidates survive the cap but rank below
                                # candidates that also share an exact token.
                                score += 5.0 * jaccard_4g
                                
                    # Add digit matching bonus to ensure candidates from address-fallback don't score 0.0
                    # and get pushed out of the MAX_CANDS cutoff.
                    s1_digits = extract_digits(s1_addrs[i])
                    c_digits = extract_digits(s23_addrs[ci])
                    shared_digits = set(s1_digits) & set(c_digits)
                    score += len(shared_digits) * 2.0
                    for d1 in s1_digits:
                        for d2 in c_digits:
                            if len(d1) >= 3 and len(d2) >= 3 and (d1.startswith(d2) or d2.startswith(d1)):
                                score += 1.0

                    scored_cands.append((score, ci))

                scored_cands.sort(key=lambda x: x[0], reverse=True)
                selected_cands = [ci for _, ci in scored_cands[:MAX_CANDS_PER_ENTITY]]
            else:
                selected_cands = list(cand_indices)

            cand_ids = [s23_ids[ci] for ci in selected_cands]
            batch_candidates_map[sid] = cand_ids
            total_candidates += len(cand_ids)

            for ci in selected_cands:
                f = compute_pair_features(
                    s1_names[i], s1_addrs[i], s1_c, sid,
                    s23_names[ci], s23_addrs[ci], s23_countries[ci], s23_ids[ci],
                    s1_name_n, s1_addr_n,
                    s23_names_norm[ci], s23_addrs_norm[ci]
                )
                max_name_sim = max(
                    f.get('name_token_sort_ratio', 0),
                    f.get('name_levenshtein', 0),
                    f.get('name_token_set_ratio', 0)
                )
                batch_pairs.append(f)
                batch_pair_s1.append(sid)
                batch_pair_cand.append(s23_ids[ci])
                batch_pair_name_sim.append(max_name_sim)

        # Model prediction for this batch
        batch_predictions = defaultdict(list)
        if batch_pairs:
            df_batch = pd.DataFrame(batch_pairs)
            scores = predict_scores(model, df_batch, feature_cols)
            for k in range(len(scores)):
                # Match condition: score >= threshold AND passes co-location name consistency guard
                if scores[k] >= threshold and batch_pair_name_sim[k] >= 0.50:
                    batch_predictions[batch_pair_s1[k]].append(batch_pair_cand[k])

        # Write rows directly to output files
        for i in range(b_start, b_end):
            sid = s1_ids[i]
            matched = batch_predictions.get(sid, [])
            cands = batch_candidates_map.get(sid, [])

            if matched:
                total_matches += 1
            else:
                total_singletons += 1

            matched_str = ','.join(matched)
            cands_str = ','.join(cands)

            out_matches_f.write(f"{sid}\t{matched_str}\n")
            out_cands_f.write(f"{sid}\t{cands_str}\n")

        elapsed_b = time.time() - t_b0
        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            print(f"[{country}] Batch {b+1}/{n_batches} done in {elapsed_b:.1f}s ({len(batch_pairs)} pairs scored)")

    # Cleanup country memory
    del s23, s23_ids, s23_names, s23_addrs, s23_names_norm, s23_addrs_norm
    del s1, s1_ids, s1_names, s1_addrs, s1_names_norm, s1_addrs_norm
    del name_idx, addr_idx, prefix_idx, bigram_idx
    gc.collect()

    t_country = time.time() - t_start
    print(f"[{country}] COMPLETED in {t_country:.1f}s ({t_country/60:.1f} min)")
    print(f"[{country}] Matches found: {total_matches} ({total_matches/n_s1*100:.1f}%), Singletons: {total_singletons} ({total_singletons/n_s1*100:.1f}%), Avg candidates: {total_candidates/n_s1:.1f}")


def main():
    t_global = time.time()
    print("=" * 80)
    print("STARTING FULL TEST INFERENCE PIPELINE (CALIBRATED FOR UNSEEN COUNTRIES)")
    print("=" * 80)

    # Load model
    print(f"Loading trained model from {MODEL_PATH}...")
    model, feature_cols, base_threshold = load_model(MODEL_PATH)
    print(f"Model loaded successfully! Invariant features: {len(feature_cols)}, Base Threshold: {base_threshold:.3f}")
    print(f"Country-specific thresholds: {COUNTRY_THRESHOLDS}")

    matches_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cands_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    with open(matches_file, 'w', encoding='utf-8') as f_m, \
         open(cands_file, 'w', encoding='utf-8') as f_c:

        # Write TSV headers
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        # Process France first (smallest), then US, then India
        countries = ["France", "US", "India"]
        for country in countries:
            process_country(country, model, feature_cols, f_m, f_c)

    print("\n" + "=" * 80)
    print(f"ALL COUNTRIES PROCESSED in {time.time()-t_global:.1f}s ({(time.time()-t_global)/60:.1f} min)")
    print("=" * 80)
    print(f"Matching results saved to: {matches_file}")
    print(f"Candidate pairs saved to:  {cands_file}")

    # Validate output
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL VALIDATOR")
    print("=" * 80)
    val_script = os.path.join(BASE_DIR, "utils", "validate_submission.py")
    cmd = f"python3 '{val_script}' --matching '{matches_file}' --candidate '{cands_file}' --test-dir '{TEST_DIR}'"
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    if ret == 0:
        print("\n✓ VALIDATION PASSED! Submission files are 100% compliant.")
    else:
        print(f"\n✗ Validation returned exit code {ret}.")


if __name__ == '__main__':
    main()
