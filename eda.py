#!/usr/bin/env python3
"""Exploratory Data Analysis for Business Entity Resolution."""

import pandas as pd
import numpy as np
import os

BASE = "/Users/daxsavaliya/Desktop/Aura_submission"
TRAIN = os.path.join(BASE, "dataset/train")
TEST = os.path.join(BASE, "dataset/test")

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load training data
train_s1 = pd.read_csv(os.path.join(TRAIN, "train_source1.tsv"), sep="\t")
train_s2 = pd.read_csv(os.path.join(TRAIN, "train_source2.tsv"), sep="\t")
train_s3 = pd.read_csv(os.path.join(TRAIN, "train_source3.tsv"), sep="\t")
train_gt = pd.read_csv(os.path.join(TRAIN, "train_ground_truth.tsv"), sep="\t")

print(f"Train S1: {train_s1.shape}")
print(f"Train S2: {train_s2.shape}")
print(f"Train S3: {train_s3.shape}")
print(f"Train GT: {train_gt.shape}")

# Load test data
test_s1 = pd.read_csv(os.path.join(TEST, "test_source1.tsv"), sep="\t")
test_s2 = pd.read_csv(os.path.join(TEST, "test_source2.tsv"), sep="\t")
test_s3 = pd.read_csv(os.path.join(TEST, "test_source3.tsv"), sep="\t")

print(f"\nTest S1: {test_s1.shape}")
print(f"Test S2: {test_s2.shape}")
print(f"Test S3: {test_s3.shape}")

print("\n" + "=" * 80)
print("COLUMN NAMES")
print("=" * 80)
for name, df in [("S1", train_s1), ("S2", train_s2), ("S3", train_s3), ("GT", train_gt)]:
    print(f"{name}: {list(df.columns)}")

print("\n" + "=" * 80)
print("SAMPLE ROWS")
print("=" * 80)
for name, df in [("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3)]:
    print(f"\n--- {name} ---")
    print(df.head(5).to_string())

print("\n--- Ground Truth ---")
print(train_gt.head(10).to_string())

print("\n" + "=" * 80)
print("DATA TYPES & NULLS")
print("=" * 80)
for name, df in [("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3)]:
    print(f"\n--- {name} ---")
    print(df.dtypes)
    print(f"Nulls:\n{df.isnull().sum()}")
    print(f"Total rows: {len(df)}")

print("\n" + "=" * 80)
print("COUNTRY DISTRIBUTIONS")
print("=" * 80)
for name, df in [("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3),
                  ("Test S1", test_s1), ("Test S2", test_s2), ("Test S3", test_s3)]:
    print(f"\n--- {name} ---")
    print(df['country'].value_counts())

print("\n" + "=" * 80)
print("GROUND TRUTH ANALYSIS")
print("=" * 80)

# Count matches per S1 entity
train_gt['match_count'] = train_gt['matched_entity_ids'].apply(
    lambda x: 0 if pd.isna(x) or str(x).strip() == '' else len(str(x).split(','))
)

print(f"\nMatch count distribution:")
print(train_gt['match_count'].value_counts().sort_index())

singletons = (train_gt['match_count'] == 0).sum()
with_matches = (train_gt['match_count'] > 0).sum()
print(f"\nSingletons (no match): {singletons} ({singletons/len(train_gt)*100:.1f}%)")
print(f"With matches: {with_matches} ({with_matches/len(train_gt)*100:.1f}%)")

# Analyze which sources are matched
def get_sources(matched_ids):
    if pd.isna(matched_ids) or str(matched_ids).strip() == '':
        return set()
    ids = str(matched_ids).split(',')
    return {i.split('-')[0] for i in ids}

train_gt['matched_sources'] = train_gt['matched_entity_ids'].apply(get_sources)
s2_only = train_gt['matched_sources'].apply(lambda x: x == {'S2'}).sum()
s3_only = train_gt['matched_sources'].apply(lambda x: x == {'S3'}).sum()
both = train_gt['matched_sources'].apply(lambda x: x == {'S2', 'S3'}).sum()
neither = train_gt['matched_sources'].apply(lambda x: x == set()).sum()

print(f"\nS2 only matches: {s2_only}")
print(f"S3 only matches: {s3_only}")
print(f"Both S2 and S3 matches: {both}")
print(f"No matches: {neither}")

# How many S2/S3 IDs per match
def count_by_source(matched_ids, prefix):
    if pd.isna(matched_ids) or str(matched_ids).strip() == '':
        return 0
    ids = str(matched_ids).split(',')
    return sum(1 for i in ids if i.startswith(prefix))

train_gt['s2_count'] = train_gt['matched_entity_ids'].apply(lambda x: count_by_source(x, 'S2'))
train_gt['s3_count'] = train_gt['matched_entity_ids'].apply(lambda x: count_by_source(x, 'S3'))

print(f"\nS2 match count distribution (among those with S2 matches):")
print(train_gt[train_gt['s2_count'] > 0]['s2_count'].value_counts().sort_index().head(10))
print(f"\nS3 match count distribution (among those with S3 matches):")
print(train_gt[train_gt['s3_count'] > 0]['s3_count'].value_counts().sort_index().head(10))

print("\n" + "=" * 80)
print("NAME PATTERNS")
print("=" * 80)

# Sample some matched pairs to understand noise
sample_gt = train_gt[train_gt['match_count'] > 0].head(20)
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    s1_row = train_s1[train_s1['entity_id'] == s1_id].iloc[0]
    matched_ids = str(row['matched_entity_ids']).split(',')
    print(f"\n  S1: [{s1_row['entity_id']}] {s1_row['business_name']} | {s1_row['business_address']} | {s1_row['country']}")
    for mid in matched_ids[:3]:
        mid = mid.strip()
        if mid.startswith('S2'):
            df = train_s2
        else:
            df = train_s3
        match_row = df[df['entity_id'] == mid]
        if len(match_row) > 0:
            mr = match_row.iloc[0]
            print(f"  -> [{mr['entity_id']}] {mr['business_name']} | {mr['business_address']} | {mr['country']}")

print("\n" + "=" * 80)
print("BUSINESS NAME LENGTH STATS")
print("=" * 80)
for name, df in [("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3)]:
    lengths = df['business_name'].astype(str).str.len()
    print(f"{name}: mean={lengths.mean():.1f}, median={lengths.median():.1f}, max={lengths.max()}, min={lengths.min()}")

print("\n" + "=" * 80)
print("ADDRESS ANALYSIS")
print("=" * 80)
for name, df in [("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3)]:
    addr_null = df['business_address'].isna().sum()
    addr_empty = (df['business_address'].astype(str).str.strip() == '').sum()
    print(f"{name}: null addresses={addr_null}, empty addresses={addr_empty}")
    lengths = df['business_address'].dropna().astype(str).str.len()
    print(f"  Length: mean={lengths.mean():.1f}, median={lengths.median():.1f}, max={lengths.max()}")

print("\n" + "=" * 80)
print("ENTITY ID FORMATS")
print("=" * 80)
print(f"S1 prefix check: all start with S1-: {train_s1['entity_id'].str.startswith('S1-').all()}")
print(f"S2 prefix check: all start with S2-: {train_s2['entity_id'].str.startswith('S2-').all()}")
print(f"S3 prefix check: all start with S3-: {train_s3['entity_id'].str.startswith('S3-').all()}")
print(f"S1 IDs unique: {train_s1['entity_id'].nunique() == len(train_s1)}")
print(f"S2 IDs unique: {train_s2['entity_id'].nunique() == len(train_s2)}")
print(f"S3 IDs unique: {train_s3['entity_id'].nunique() == len(train_s3)}")

# Check test
print(f"\nTest S1 IDs: {test_s1['entity_id'].nunique()}")
print(f"Test S2 IDs: {test_s2['entity_id'].nunique()}")
print(f"Test S3 IDs: {test_s3['entity_id'].nunique()}")

print("\nDONE.")
