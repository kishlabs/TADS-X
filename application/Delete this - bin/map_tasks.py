"""
map_tasks.py — run on your machine
Reads all 14 task_N_train.json files and shows the top preferred classes,
so we can map file numbers to actual task names.
"""
import json, os
from collections import Counter

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

tasks_dir = r"E:\DVCon\COCO\dataset-master\coco-tasks\annotations"

print(f"{'File':<8} {'#pref':>6}  Top 5 preferred classes")
print("-" * 75)

for n in range(1, 15):
    path = os.path.join(tasks_dir, f"task_{n}_train.json")
    with open(path) as f:
        data = json.load(f)
    anns = data["annotations"]
    preferred = [a for a in anns if a.get("category_id") == 1]
    counts = Counter()
    for a in preferred:
        idx = COCO_ID_TO_IDX.get(a["COCO_category_id"])
        if idx is not None:
            counts[COCO_CLASSES[idx]] += 1
    top5 = ", ".join(f"{cls}({cnt})" for cls, cnt in counts.most_common(5))
    print(f"task_{n:<2}  {len(preferred):>6}  {top5}")
