"""
inspect_annotation.py
Run this FIRST to see the exact JSON structure of COCO-Tasks annotations.
Usage: python inspect_annotation.py
"""
import json

path = r"E:\DVCon\COCO\dataset-master\coco-tasks\annotations\task_1_train.json"

with open(path, "r") as f:
    data = json.load(f)

print("=== Top-level type ===")
if isinstance(data, dict):
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  '{k}': list of {len(v)} items")
        else:
            print(f"  '{k}': {type(v).__name__} = {str(v)[:80]}")
elif isinstance(data, list):
    print(f"  Root is a list of {len(data)} items")

print("\n=== First 5 annotations (full content) ===")
anns = data.get("annotations", data) if isinstance(data, dict) else data
for i, ann in enumerate(anns[:5]):
    print(f"\n--- Annotation {i} ---")
    for k, v in ann.items():
        print(f"  {k}: {v}")

print("\n=== Unique keys across first 50 annotations ===")
all_keys = set()
for ann in anns[:50]:
    all_keys.update(ann.keys())
print(f"  {sorted(all_keys)}")

print("\n=== Unique values of suspicious fields (first 200 anns) ===")
candidates = ['label', 'correct', 'suitability', 'answer', 'suitable',
              'most_suitable', 'ground_truth', 'score', 'suitable_object']
for field in candidates:
    vals = set()
    for ann in anns[:200]:
        if field in ann:
            vals.add(ann[field])
    if vals:
        print(f"  '{field}': {vals}")
