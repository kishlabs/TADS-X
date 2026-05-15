import numpy as np
import os
import sys

# Change working directory or add to path to import task_definitions
sys.path.append('e:\\DVCon\\application')
A = np.load('e:/DVCon/application/data/affordance_matrix.npy')  # (14, 80)

from task_definitions import PAPER_TASKS, COCO_CLASSES

# Classes of interest
check = {
    'book':      73,
    'clock':     74,
    'bed':       59,
    'couch':     57,
    'suitcase':  28,
    'backpack':  24,
    'handbag':   26,
    'scissors':  76,
    'cell phone': 67,
    'laptop':    63,
    'wine glass': 40,
    'cup':       41,
    'bowl':      45,
}

print(f"{'Class':<15} {'Best Paper Task':<50} {'Score'}")
print("-" * 80)
for cls_name, cls_idx in check.items():
    col = A[:, cls_idx]
    best_row = col.argmax()
    print(f"{cls_name:<15} Task {best_row+1:2d}: {PAPER_TASKS[best_row+1]:<45} {col[best_row]:.4f}")
