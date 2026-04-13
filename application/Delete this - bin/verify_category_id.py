"""
verify_category_id.py  — run on your machine
Checks what category_id=0 vs category_id=1 means in COCO-Tasks.
"""
import json
from collections import Counter

path = r"E:\DVCon\COCO\dataset-master\coco-tasks\annotations\task_1_train.json"

with open(path) as f:
    data = json.load(f)

anns = data["annotations"]

# Count all category_id values
cat_id_counts = Counter(ann["category_id"] for ann in anns)
print("=== category_id value distribution ===")
for val, count in sorted(cat_id_counts.items()):
    print(f"  category_id={val}: {count} annotations")

# Show the 'categories' list  
print("\n=== categories field ===")
for cat in data.get("categories", []):
    print(f"  {cat}")

# Show a few annotations where category_id == 1
print("\n=== 3 annotations where category_id == 1 (should be correct answer) ===")
positives = [a for a in anns if a["category_id"] == 1]
print(f"  Total positives: {len(positives)}")

# COCO class names for reference
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]
_COCO_CAT_IDS = [
    1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
    22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,
    43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
    62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,
    85,86,87,88,89,90
]
COCO_ID_TO_IDX = {cat_id: idx for idx, cat_id in enumerate(_COCO_CAT_IDS)}

for ann in positives[:5]:
    coco_id = ann["COCO_category_id"]
    idx = COCO_ID_TO_IDX.get(coco_id, "?")
    name = COCO_CLASSES[idx] if isinstance(idx, int) else "unknown"
    print(f"  image_id={ann['image_id']}  COCO_category_id={coco_id} ({name})")

# Count top classes among positives for task 1 (expect wine glass to dominate)
print("\n=== Top 10 COCO classes among positives (task 1 = serve wine) ===")
class_counts = Counter()
for ann in positives:
    coco_id = ann["COCO_category_id"]
    idx = COCO_ID_TO_IDX.get(coco_id)
    if idx is not None:
        class_counts[COCO_CLASSES[idx]] += 1

for cls, cnt in class_counts.most_common(10):
    print(f"  {cls:<20} {cnt}")
