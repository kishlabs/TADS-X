import torch
from embeddings import load_projected_embeddings
from pipeline import resolve_task_id
from task_definitions import SRS_TASKS

cache, _ = load_projected_embeddings(
    'data/task_raw_embeddings.pt',
    'data/projection_layer_trained.pt'
)

print('SRS Task → Paper Task mapping:')
seen = {}
for srs_id, srs_str in SRS_TASKS.items():
    try:
        r = resolve_task_id(srs_str, cache)
        flag = ' ← COLLISION' if r.paper_task_id_1 in seen.values() else ''
        seen[srs_id] = r.paper_task_id_1
        print(f'  SRS {srs_id:2d} "{srs_str:<22}" → paper {r.paper_task_id_1:2d} "{r.paper_task_str}"  sim={r.cosine_sim:.4f}{flag}')
    except KeyError as e:
        print(f'  SRS {srs_id:2d} "{srs_str:<22}" → BLOCKED (sim < 0.3)')
