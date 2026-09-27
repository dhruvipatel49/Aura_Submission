#!/usr/bin/env python3
"""
Manual Spot-Check Script:
Samples 10 random predictions for US, 10 for India, 10 for France
from final output/matching_results.tsv and prints side-by-side comparisons.
"""

import os, sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

print("Loading test sources and matching results...")
s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep='\t')
s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep='\t')
s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep='\t')
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3

s23_map = {row['entity_id']: (str(row['business_name']), str(row['business_address']))
           for _, row in s23.iterrows()}

matches_df = pd.read_csv(os.path.join(OUTPUT_DIR, "matching_results.tsv"), sep='\t')
matches_map = {row['source1_entity_id']: str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
               for _, row in matches_df.iterrows()}

np.random.seed(42)

for country in ["France", "US", "India"]:
    print("\n" + "=" * 90)
    print(f"SPOT CHECK: 10 SAMPLE PREDICTIONS FOR {country.upper()}")
    print("=" * 90)
    
    country_s1 = s1[s1['country'] == country].sample(n=10, random_state=42).reset_index(drop=True)
    
    for idx, row in country_s1.iterrows():
        sid = row['entity_id']
        s1_name = str(row['business_name'])
        s1_addr = str(row['business_address'])
        matched_str = matches_map.get(sid, '')
        
        print(f"\n[{country} #{idx+1:02d}] S1 ID: {sid}")
        print(f"  S1 Record:      Name:    '{s1_name}'")
        print(f"                  Address: '{s1_addr}'")
        
        if not matched_str or matched_str.strip() == '':
            print("  Prediction:     [SINGLETON (No Match Predicted)]")
        else:
            m_ids = [m.strip() for m in matched_str.split(',') if m.strip()]
            print(f"  Matches ({len(m_ids)}):")
            for m_idx, mid in enumerate(m_ids):
                c_info = s23_map.get(mid, ("N/A", "N/A"))
                print(f"    ({m_idx+1}) ID: {mid}")
                print(f"        Name:    '{c_info[0]}'")
                print(f"        Address: '{c_info[1]}'")

