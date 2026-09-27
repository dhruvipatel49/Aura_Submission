#!/usr/bin/env python3
"""
Blocking / Candidate Generation for Business Entity Resolution.
Optimized for 10M+ records using vectorized operations.

Multi-strategy blocking:
1. TF-IDF char n-gram + ANN on name+addr (primary recall driver)
2. Token-inverted-index on normalized names (fast, high precision)
3. Country hard-filter (always applied)
4. Lightweight tightening pass (Jaccard threshold)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (
    normalize_name, normalize_address, extract_digits,
    is_non_latin, strip_accents, get_name_tokens
)


def _normalize_names_vec(names):
    """Vectorized name normalization via apply."""
    return [normalize_name(str(n)) for n in names]


def _normalize_addrs_vec(addrs):
    """Vectorized address normalization via apply."""
    return [normalize_address(str(a)) for a in addrs]


def _make_tfidf_text(names_norm, addrs_norm):
    """Combine normalized name + address for TF-IDF."""
    return [f"{n} {a}".strip() for n, a in zip(names_norm, addrs_norm)]


def build_token_inverted_index(entity_ids, names_norm):
    """
    Build inverted index: token -> set of entity indices.
    Uses normalized name tokens. Vectorized-ish via list comprehension.
    """
    token_to_indices = defaultdict(list)
    for idx, name in enumerate(names_norm):
        if not name:
            continue
        tokens = set(name.split())
        for tok in tokens:
            if len(tok) >= 3:
                token_to_indices[tok].append(idx)

    # Convert to arrays for efficiency
    for tok in token_to_indices:
        token_to_indices[tok] = np.array(token_to_indices[tok], dtype=np.int32)

    return token_to_indices


def token_blocking_lookup(s1_names_norm, s23_token_index, max_bucket=5000):
    """
    For each S1 entity, find S2/S3 candidates sharing name tokens.
    Returns list of sets of s23 indices.
    """
    candidates = [set() for _ in range(len(s1_names_norm))]

    for i, name in enumerate(s1_names_norm):
        if not name:
            continue
        tokens = set(name.split())
        for tok in tokens:
            if len(tok) < 3:
                continue
            bucket = s23_token_index.get(tok)
            if bucket is not None and len(bucket) <= max_bucket:
                candidates[i].update(bucket.tolist())

    return candidates


def tfidf_blocking(s1_texts, s23_texts, n_neighbors=10, ngram_range=(2, 4),
                   max_features=50000, distance_threshold=0.80):
    """
    TF-IDF char n-gram blocking using sklearn NearestNeighbors.
    Returns list of sets of s23 indices (one per S1 entity).
    """
    print(f"  Building TF-IDF (ngram={ngram_range}, max_feat={max_features})...")
    t0 = time.time()

    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )

    print(f"    Fitting on {len(s23_texts)} S2/S3 texts...")
    s23_vecs = vectorizer.fit_transform(s23_texts)
    print(f"    Matrix shape: {s23_vecs.shape}, nnz: {s23_vecs.nnz}")

    print(f"    Transforming {len(s1_texts)} S1 texts...")
    s1_vecs = vectorizer.transform(s1_texts)

    print(f"    NearestNeighbors (n={n_neighbors}, metric=cosine)...")
    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors, s23_vecs.shape[0]),
        metric='cosine',
        algorithm='brute',
        n_jobs=-1,
    )
    nn.fit(s23_vecs)

    print(f"    Querying...")
    distances, indices = nn.kneighbors(s1_vecs)

    # Build candidate sets with distance threshold
    candidates = []
    for i in range(len(s1_texts)):
        cand_set = set()
        for j_idx, dist in zip(indices[i], distances[i]):
            if dist < distance_threshold:
                cand_set.add(int(j_idx))
        candidates.append(cand_set)

    elapsed = time.time() - t0
    sizes = [len(c) for c in candidates]
    print(f"    TF-IDF done in {elapsed:.1f}s, mean candidates: {np.mean(sizes):.1f}")

    return candidates


def run_blocking(s1_df, s23_df, n_neighbors=10, use_tfidf=True):
    """
    Full multi-strategy blocking pipeline.
    Returns dict: s1_entity_id -> set of s23_entity_ids.
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"BLOCKING: {len(s1_df)} S1 × {len(s23_df)} S2/S3")
    print(f"{'='*60}")

    s1_ids = s1_df['entity_id'].values
    s23_ids = s23_df['entity_id'].values
    s1_countries = s1_df['country'].values
    s23_countries = s23_df['country'].values

    # Pre-normalize
    print("\n[1] Normalizing names and addresses...")
    s1_names_norm = _normalize_names_vec(s1_df['business_name'].values)
    s23_names_norm = _normalize_names_vec(s23_df['business_name'].values)
    s1_addrs_norm = _normalize_addrs_vec(s1_df['business_address'].values)
    s23_addrs_norm = _normalize_addrs_vec(s23_df['business_address'].values)
    print(f"    Done ({time.time()-t0:.1f}s)")

    # Build country index for fast lookup
    print("\n[2] Building country index...")
    s23_country_to_indices = defaultdict(set)
    for idx, country in enumerate(s23_countries):
        s23_country_to_indices[country].add(idx)
    print(f"    Countries: {list(s23_country_to_indices.keys())}")

    # Strategy 1: Token inverted index
    print("\n[3] Building token inverted index...")
    s23_token_index = build_token_inverted_index(s23_ids, s23_names_norm)
    print(f"    {len(s23_token_index)} unique tokens")
    token_candidates = token_blocking_lookup(s1_names_norm, s23_token_index)
    sizes = [len(c) for c in token_candidates]
    print(f"    Token blocking: mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f} candidates/entity")

    # Strategy 2: TF-IDF blocking
    if use_tfidf:
        print("\n[4] TF-IDF blocking...")
        s1_texts = _make_tfidf_text(s1_names_norm, s1_addrs_norm)
        s23_texts = _make_tfidf_text(s23_names_norm, s23_addrs_norm)
        tfidf_candidates = tfidf_blocking(
            s1_texts, s23_texts,
            n_neighbors=n_neighbors,
            distance_threshold=0.80,
        )
    else:
        tfidf_candidates = [set() for _ in range(len(s1_ids))]

    # Merge strategies
    print("\n[5] Merging and filtering...")
    merged_candidates = {}
    total_before = 0
    total_after = 0

    for i in tqdm(range(len(s1_ids)), desc="Merging"):
        s1_id = s1_ids[i]
        s1_country = s1_countries[i]

        # Union of all strategies (as s23 indices)
        all_cands = token_candidates[i] | tfidf_candidates[i]
        total_before += len(all_cands)

        # Country filter (hard - same country only)
        valid_country_indices = s23_country_to_indices.get(s1_country, set())
        filtered = all_cands & valid_country_indices
        total_after += len(filtered)

        # Convert indices to entity IDs
        if filtered:
            merged_candidates[s1_id] = {s23_ids[j] for j in filtered}
        else:
            merged_candidates[s1_id] = set()

    print(f"    Before country filter: {total_before} total pairs")
    print(f"    After country filter: {total_after} total pairs")
    cand_sizes = [len(v) for v in merged_candidates.values()]
    print(f"    Mean candidates/entity: {np.mean(cand_sizes):.1f}")
    print(f"    Median candidates/entity: {np.median(cand_sizes):.0f}")

    elapsed = time.time() - t0
    print(f"\n  Blocking completed in {elapsed:.1f}s")

    return merged_candidates


def tighten_candidates(candidates, s1_df, s23_df, min_combined_sim=0.10):
    """
    Tightening pass: remove candidates with very low similarity.
    Uses fast token Jaccard on name + address overlap.
    """
    from rapidfuzz import fuzz
    print(f"\n  Tightening candidates (min_combined_sim={min_combined_sim})...")

    s1_map = {}
    for _, row in s1_df.iterrows():
        s1_map[row['entity_id']] = {
            'name': normalize_name(str(row['business_name'])),
            'addr': normalize_address(str(row['business_address'])),
            'addr_digits': set(extract_digits(str(row['business_address']))),
        }

    s23_map = {}
    for _, row in s23_df.iterrows():
        s23_map[row['entity_id']] = {
            'name': normalize_name(str(row['business_name'])),
            'addr': normalize_address(str(row['business_address'])),
            'addr_digits': set(extract_digits(str(row['business_address']))),
            'is_non_latin': is_non_latin(str(row['business_name'])),
        }

    tightened = {}
    kept_total = 0
    removed_total = 0

    for s1_id, cand_set in tqdm(candidates.items(), desc="Tightening"):
        s1_info = s1_map.get(s1_id)
        if not s1_info:
            tightened[s1_id] = cand_set
            continue

        s1_name_tokens = set(s1_info['name'].split()) if s1_info['name'] else set()
        s1_addr_tokens = set(s1_info['addr'].split()) if s1_info['addr'] else set()

        kept = set()
        for cid in cand_set:
            c_info = s23_map.get(cid)
            if not c_info:
                continue

            c_name_tokens = set(c_info['name'].split()) if c_info['name'] else set()
            c_addr_tokens = set(c_info['addr'].split()) if c_info['addr'] else set()

            # Name token Jaccard
            if s1_name_tokens and c_name_tokens:
                name_jaccard = len(s1_name_tokens & c_name_tokens) / len(s1_name_tokens | c_name_tokens)
            else:
                name_jaccard = 0

            # Address token Jaccard
            if s1_addr_tokens and c_addr_tokens:
                addr_jaccard = len(s1_addr_tokens & c_addr_tokens) / len(s1_addr_tokens | c_addr_tokens)
            else:
                addr_jaccard = 0

            # Digit overlap
            digit_overlap = len(s1_info['addr_digits'] & c_info['addr_digits'])

            # Combined score
            combined = 0.6 * name_jaccard + 0.4 * addr_jaccard

            # Keep if reasonable OR transliteration case (non-latin) with any address signal
            if combined >= min_combined_sim or digit_overlap >= 2:
                kept.add(cid)
            elif c_info['is_non_latin'] and (addr_jaccard >= 0.12 or digit_overlap >= 1):
                kept.add(cid)

        kept_total += len(kept)
        removed_total += len(cand_set) - len(kept)
        tightened[s1_id] = kept

    print(f"    Kept: {kept_total}, Removed: {removed_total}")
    return tightened


def evaluate_blocking_recall(candidates, ground_truth_df):
    """
    Evaluate blocking recall: what fraction of true matches are in the candidate set?
    """
    total_true = 0
    found = 0
    missed_count = 0

    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched_ids = row['matched_entity_ids']
        if pd.isna(matched_ids) or str(matched_ids).strip() == '':
            continue

        true_matches = set(str(matched_ids).split(','))
        total_true += len(true_matches)

        cand_set = candidates.get(s1_id, set())
        found += len(true_matches & cand_set)
        if len(true_matches - cand_set) > 0:
            missed_count += 1

    recall = found / total_true if total_true > 0 else 0
    print(f"\n  Blocking Recall: {recall:.4f} ({found}/{total_true})")
    print(f"  Entities with missed matches: {missed_count}")
    return recall
