"""
final_mapping.py
Cross-reference the COCO-Tasks file numbers against the paper's task list
by checking which SRS task name best fits each file's preferred-class distribution.

Run on your machine:
    python final_mapping.py
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
_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,
        31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,
        55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,
        82,84,85,86,87,88,89,90]
ID2IDX = {c: i for i, c in enumerate(_IDS)}

# Classes that are strong signals for each SRS task
TASK_SIGNALS = {
    "serve wine":         {"wine glass", "cup"},
    "pour water into":    {"cup", "bottle", "bowl", "wine glass"},
    "cut something with": {"knife", "scissors", "fork"},
    "hit something with": {"baseball bat", "tennis racket", "sports ball"},
    "dig a hole with":    {"fork", "knife", "spoon"},
    "scoop something with": {"spoon", "fork", "bowl"},
    "pound something with": {"baseball bat", "bottle"},
    "cool something in":  {"refrigerator", "bowl", "cup"},
    "sit on":             {"chair", "couch", "bench", "toilet"},
    "lie on":             {"bed", "couch", "bench"},
    "carry things in":    {"backpack", "handbag", "suitcase", "bottle"},
    "read":               {"book", "laptop", "cell phone"},
    "check the time with":{"clock", "cell phone"},
    "look through":       {"tv", "laptop", "cell phone", "skis"},
}

d = r"E:\DVCon\COCO\dataset-master\coco-tasks\annotations"

print(f"{'File':<8} {'#pref':>5}  {'Top 3 classes':<45}  Best signal match")
print("-" * 90)

for n in range(1, 15):
    path = os.path.join(d, f"task_{n}_train.json")
    data = json.load(open(path))
    pref = [a for a in data["annotations"] if a.get("category_id") == 1]
    ctr = Counter(
        COCO_CLASSES[ID2IDX[a["COCO_category_id"]]]
        for a in pref if a["COCO_category_id"] in ID2IDX
    )
    top3 = ", ".join(f"{c}({v})" for c, v in ctr.most_common(3))
    top_set = {c for c, _ in ctr.most_common(5)}

    # Score each task by how many signal classes appear in top-10
    top10_set = {c for c, _ in ctr.most_common(10)}
    scores = {task: len(sigs & top10_set) for task, sigs in TASK_SIGNALS.items()}
    best = max(scores, key=scores.get)
    best_score = scores[best]

    print(f"task_{n:<2}  {len(pref):>5}  {top3:<45}  {best} ({best_score} signals)")
