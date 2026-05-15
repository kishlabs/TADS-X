"""
task_definitions.py
===================
TADS-X — Team ChipSmiths | DVCon India 2026

Single source of truth for all task-related constants.

IMPORTANT ARCHITECTURAL NOTE:
------------------------------
The COCO-Tasks paper (Sawatzky et al. 2019) defines 14 tasks with specific
natural-language descriptions (e.g., "step on something to reach top of a shelf").
These are the tasks the model is TRAINED on, and their file numbering (1-14) is
what indexes the affordance matrix rows and task embedding cache.

The DVCon India 2026 contest defines 14 EVALUATION QUERIES
(e.g., "serve wine", "pour water into") which are semantically related but NOT
identical to the paper's task strings.

At inference time, the pipeline resolves a contest query to the nearest
paper task via cosine similarity in the TinyBERT embedding space.
This is INTENTIONAL: it generalises the model beyond the exact training vocabulary.

Do NOT create a hardcoded FILE_ID → SRS_ID mapping — the mapping is learned,
not hardcoded, and is handled by nearest-neighbour lookup in pipeline.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# PAPER TASKS  (Sawatzky et al. 2019, Table 1)
# These are the training task strings. They index the affordance matrix rows
# (0-indexed: row 0 = FILE_ID 1 = "step on something...").
# The annotation files are task_N_train.json where N = FILE_ID.
# ─────────────────────────────────────────────────────────────────────────────

PAPER_TASKS = {
    1:  "step on something to reach top of a shelf",
    2:  "sit comfortably",
    3:  "place flowers",
    4:  "get potatoes out of fire",
    5:  "water plant",
    6:  "get lemon out of tea",
    7:  "dig hole",
    8:  "open bottle of beer",
    9:  "open parcel",
    10: "serve wine",
    11: "pour sugar",
    12: "smear butter",
    13: "extinguish fire",
    14: "pound carpet",
}

# Convenience: 0-indexed list in file order (for matrix row access)
PAPER_TASK_LIST = [PAPER_TASKS[i] for i in range(1, 15)]  # index 0 = file task 1


# ─────────────────────────────────────────────────────────────────────────────
# SRS / CONTEST TASKS  (DVCon India 2026 problem statement)
# These are the 14 evaluation queries the contest will use at test time.
# The pipeline resolves these to the nearest PAPER_TASK at inference.
# ─────────────────────────────────────────────────────────────────────────────

SRS_TASKS = {
    1:  "serve wine",
    2:  "pour water into",
    3:  "cut something with",
    4:  "hit something with",
    5:  "dig a hole with",
    6:  "scoop something with",
    7:  "pound something with",
    8:  "cool something in",
    9:  "sit on",
    10: "lie on",
    11: "carry things in",
    12: "read",
    13: "check the time with",
    14: "look through",
}

SRS_TASK_LIST = [SRS_TASKS[i] for i in range(1, 15)]


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN APPROXIMATE MAPPING  (informational only — NOT used in code)
# Derived from annotation class distribution analysis.
# The pipeline uses learned cosine similarity — this table is for documentation.
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
_APPROX_PAPER_TO_SRS = {
    1:  9,   # "step on something..."     → "sit on"
    2:  10,  # "sit comfortably"          → "lie on"
    3:  11,  # "place flowers"            → "carry things in"
    4:  4,   # "get potatoes out of fire" → "hit something with"
    5:  2,   # "water plant"              → "pour water into"
    6:  6,   # "get lemon out of tea"     → "scoop something with"
    7:  5,   # "dig hole"                 → "dig a hole with"
    8:  None,# "open bottle of beer"      → no clean SRS match
    9:  3,   # "open parcel"              → "cut something with"
    10: 1,   # "serve wine"               → "serve wine"
    11: 2,   # "pour sugar"               → "pour water into"
    12: 6,   # "smear butter"             → "scoop something with"
    13: 8,   # "extinguish fire"          → "cool something in"
    14: 7,   # "pound carpet"             → "pound something with"
}
# fmt: on


# ─────────────────────────────────────────────────────────────────────────────
# COCO 80-class constants shared across all modules
# ─────────────────────────────────────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# COCO official category_id → 0-based matrix index (gaps in 1..90)
_COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
COCO_ID_TO_IDX = {cat_id: idx for idx, cat_id in enumerate(_COCO_CAT_IDS)}
IDX_TO_CLASS   = {idx: name for idx, name in enumerate(COCO_CLASSES)}
NUM_CLASSES    = 80
NUM_TASKS      = 14
WORKING_DIM = 256   # TinyBERT projection output dim — shared by TCFG, AGCA, SCRN, pipeline
