#!/usr/bin/env python3
"""
Feature engineering for Business Entity Resolution.
Computes rich similarity features for each (S1, candidate) pair.
Optimized with dict-based lookups instead of DataFrame iterrows.
"""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (
    normalize_name, normalize_address, extract_digits,
    is_non_latin, normalize_name_sorted_tokens
)


def _token_jaccard(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def _token_overlap(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1)


def compute_pair_features(s1_name, s1_addr, s1_country, s1_eid,
                          c_name, c_addr, c_country, c_eid,
                          s1_name_norm, s1_addr_norm,
                          c_name_norm, c_addr_norm):
    """
    Compute features for a single (S1, candidate) pair.
    Accepts pre-normalized values for speed.
    """
    features = {}

    s1_name_tokens = set(s1_name_norm.split()) if s1_name_norm else set()
    c_name_tokens = set(c_name_norm.split()) if c_name_norm else set()
    s1_addr_tokens = set(s1_addr_norm.split()) if s1_addr_norm else set()
    c_addr_tokens = set(c_addr_norm.split()) if c_addr_norm else set()

    # ---- Name features ----
    if s1_name_norm and c_name_norm:
        features['name_levenshtein'] = fuzz.ratio(s1_name_norm, c_name_norm) / 100.0
        features['name_token_sort_ratio'] = fuzz.token_sort_ratio(s1_name_norm, c_name_norm) / 100.0
        features['name_token_set_ratio'] = fuzz.token_set_ratio(s1_name_norm, c_name_norm) / 100.0
        features['name_partial_ratio'] = fuzz.partial_ratio(s1_name_norm, c_name_norm) / 100.0
        features['name_jaro_winkler'] = JaroWinkler.similarity(s1_name_norm, c_name_norm)
    else:
        features['name_levenshtein'] = 0.0
        features['name_token_sort_ratio'] = 0.0
        features['name_token_set_ratio'] = 0.0
        features['name_partial_ratio'] = 0.0
        features['name_jaro_winkler'] = 0.0

    features['name_token_jaccard'] = _token_jaccard(s1_name_tokens, c_name_tokens)
    features['name_token_overlap_s1'] = _token_overlap(s1_name_tokens, c_name_tokens)
    features['name_token_overlap_cand'] = _token_overlap(c_name_tokens, s1_name_tokens)
    features['name_exact_match'] = 1.0 if s1_name_norm == c_name_norm and s1_name_norm else 0.0

    # Sorted token match
    s1_sorted = ' '.join(sorted(s1_name_tokens)) if s1_name_tokens else ''
    c_sorted = ' '.join(sorted(c_name_tokens)) if c_name_tokens else ''
    features['name_sorted_exact_match'] = 1.0 if s1_sorted == c_sorted and s1_sorted else 0.0

    # Structural
    features['name_len_s1'] = len(s1_name_norm)
    features['name_len_cand'] = len(c_name_norm)
    features['name_len_diff'] = abs(len(s1_name_norm) - len(c_name_norm))
    max_len = max(len(s1_name_norm), len(c_name_norm))
    features['name_len_ratio'] = min(len(s1_name_norm), len(c_name_norm)) / max_len if max_len > 0 else 0.0
    features['name_token_count_s1'] = len(s1_name_tokens)
    features['name_token_count_cand'] = len(c_name_tokens)
    features['name_token_count_diff'] = abs(len(s1_name_tokens) - len(c_name_tokens))

    # Non-Latin detection
    s1_non_latin = 1.0 if is_non_latin(s1_name) else 0.0
    c_non_latin = 1.0 if is_non_latin(c_name) else 0.0
    features['s1_name_is_non_latin'] = s1_non_latin
    features['cand_name_is_non_latin'] = c_non_latin
    features['names_diff_script'] = 1.0 if s1_non_latin != c_non_latin else 0.0

    # ---- Address features ----
    if s1_addr_norm and c_addr_norm:
        features['addr_levenshtein'] = fuzz.ratio(s1_addr_norm, c_addr_norm) / 100.0
        features['addr_token_sort_ratio'] = fuzz.token_sort_ratio(s1_addr_norm, c_addr_norm) / 100.0
        features['addr_token_set_ratio'] = fuzz.token_set_ratio(s1_addr_norm, c_addr_norm) / 100.0
    else:
        features['addr_levenshtein'] = 0.0
        features['addr_token_sort_ratio'] = 0.0
        features['addr_token_set_ratio'] = 0.0

    features['addr_token_jaccard'] = _token_jaccard(s1_addr_tokens, c_addr_tokens)
    features['addr_token_overlap_s1'] = _token_overlap(s1_addr_tokens, c_addr_tokens)
    features['addr_token_overlap_cand'] = _token_overlap(c_addr_tokens, s1_addr_tokens)

    # Digit matching
    s1_digits = set(extract_digits(s1_addr))
    c_digits = set(extract_digits(c_addr))
    features['addr_digit_overlap'] = len(s1_digits & c_digits)
    union_digits = s1_digits | c_digits
    features['addr_digit_jaccard'] = len(s1_digits & c_digits) / len(union_digits) if union_digits else 0.0

    # Postal code / street number exact match (strong signal)
    # Extract the longest digit sequence as a proxy for zip/postal code
    s1_long_digits = [d for d in s1_digits if len(d) >= 4]
    c_long_digits = [d for d in c_digits if len(d) >= 4]
    shared_long = set(s1_long_digits) & set(c_long_digits)
    features['addr_postal_exact'] = 1.0 if shared_long else 0.0
    features['addr_has_postal'] = 1.0 if (s1_long_digits or c_long_digits) else 0.0

    # Numbers in business name (branch numbers, IDs)
    s1_name_digits = set(extract_digits(s1_name))
    c_name_digits = set(extract_digits(c_name))
    union_name_digits = s1_name_digits | c_name_digits
    features['name_digit_jaccard'] = (len(s1_name_digits & c_name_digits) / len(union_name_digits)
                                      if union_name_digits else 0.0)
    features['name_has_digits'] = 1.0 if (s1_name_digits or c_name_digits) else 0.0
    features['name_digits_exact'] = 1.0 if (s1_name_digits and s1_name_digits == c_name_digits) else 0.0

    features['s1_has_addr'] = 1.0 if s1_addr_norm else 0.0
    features['cand_has_addr'] = 1.0 if c_addr_norm else 0.0
    features['both_have_addr'] = 1.0 if (s1_addr_norm and c_addr_norm) else 0.0

    max_addr_len = max(len(s1_addr_norm), len(c_addr_norm))
    features['addr_len_diff'] = abs(len(s1_addr_norm) - len(c_addr_norm))
    features['addr_len_ratio'] = min(len(s1_addr_norm), len(c_addr_norm)) / max_addr_len if max_addr_len > 0 else 0.0

    # ---- Country & combined ----
    features['country_match'] = 1.0 if s1_country == c_country else 0.0

    features['max_name_sim'] = max(
        features['name_token_sort_ratio'],
        features['name_token_set_ratio'],
        features['name_token_jaccard'],
        features['name_jaro_winkler'],
    )
    features['max_addr_sim'] = max(
        features['addr_token_sort_ratio'],
        features['addr_token_set_ratio'],
        features['addr_token_jaccard']
    )
    features['combined_sim'] = 0.6 * features['max_name_sim'] + 0.4 * features['max_addr_sim']

    features['is_s2'] = 1.0 if str(c_eid).startswith('S2-') else 0.0
    features['is_s3'] = 1.0 if str(c_eid).startswith('S3-') else 0.0

    return features


def compute_features_batch(s1_df, s23_df, candidates):
    """
    Compute features for all (S1, candidate) pairs.
    Uses dict lookups for speed — no iterrows in the hot loop.
    Only normalizes S2/S3 entities that actually appear as candidates.
    """
    needed_cand_ids = set()
    for c_set in candidates.values():
        needed_cand_ids.update(c_set)

    # Build lookup dicts using fast zip
    s1_data = {}
    for eid, name, addr, country in zip(s1_df['entity_id'], s1_df['business_name'], s1_df['business_address'], s1_df['country']):
        s1_data[eid] = {
            'name': str(name),
            'addr': str(addr),
            'country': str(country),
            'name_norm': normalize_name(str(name)),
            'addr_norm': normalize_address(str(addr)),
        }

    s23_data = {}
    s23_needed = s23_df[s23_df['entity_id'].isin(needed_cand_ids)]
    for eid, name, addr, country in zip(s23_needed['entity_id'], s23_needed['business_name'], s23_needed['business_address'], s23_needed['country']):
        s23_data[eid] = {
            'name': str(name),
            'addr': str(addr),
            'country': str(country),
            'name_norm': normalize_name(str(name)),
            'addr_norm': normalize_address(str(addr)),
        }

    total_pairs = sum(len(v) for v in candidates.values())

    all_features = []
    for s1_id, cand_set in candidates.items():
        s1_info = s1_data.get(s1_id)
        if not s1_info:
            continue

        for cand_id in cand_set:
            c_info = s23_data.get(cand_id)
            if not c_info:
                continue

            feats = compute_pair_features(
                s1_info['name'], s1_info['addr'], s1_info['country'], s1_id,
                c_info['name'], c_info['addr'], c_info['country'], cand_id,
                s1_info['name_norm'], s1_info['addr_norm'],
                c_info['name_norm'], c_info['addr_norm'],
            )
            feats['s1_id'] = s1_id
            feats['cand_id'] = cand_id
            all_features.append(feats)

    df = pd.DataFrame(all_features)
    return df
