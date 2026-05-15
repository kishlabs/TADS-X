import numpy as np
import os
import sys

sys.path.append('e:\\DVCon\\application')
from task_definitions import PAPER_TASKS

A = np.load('e:/DVCon/application/data/affordance_matrix.npy')  # (14, 80)

# Check specific classes across ALL 14 tasks
target_classes = {
    'book': 73, 'clock': 74, 'cell_phone': 67,
    'laptop': 63, 'tv': 62, 'scissors': 76,
}

for cls_name, cls_idx in target_classes.items():
    print(f"\n{cls_name} (idx {cls_idx}) across all 14 tasks:")
    for task_id in range(1, 15):
        score = A[task_id-1, cls_idx]
        if score > 0.001:  # only show non-negligible scores
            print(f"  Task {task_id:2d} ({PAPER_TASKS[task_id]:<45}): {score:.4f}")
