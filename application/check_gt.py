import torch
import os
from train import COCOTasksDataset

def main():
    print("Checking train cache GT presence...")
    try:
        train_dataset = COCOTasksDataset("data/roi_cache/train", split="train")
    except Exception as e:
        print("Error loading dataset:", e)
        return

    gt_in_candidates = 0
    total = 0
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        gt_idx = sample.get('gt_index')
        if gt_idx is not None and gt_idx >= 0:
            gt_in_candidates += 1
        total += 1
    
    if total == 0:
        print("No samples found.")
        return

    ratio = gt_in_candidates / total
    print(f"GT in candidates: {gt_in_candidates}/{total} = {ratio:.3f}")

if __name__ == "__main__":
    main()
