"""
read_readme.py — run on your machine
Reads the COCO-Tasks README to get the official task-to-file mapping.
"""
path = r"E:\DVCon\COCO\dataset-master\coco-tasks\README.md"
with open(path, encoding="utf-8") as f:
    print(f.read())
