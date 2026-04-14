"""
embeddings.py
=============
TADS-X — Team ChipSmiths | DVCon India 2026

Generates and caches task embeddings for all 14 PAPER tasks (COCO-Tasks dataset).

Design (fixes stale-cache bug from v1):
  - Saves RAW 312-D TinyBERT CLS vectors to disk (these never change)
  - Projection (312→256) is saved separately as its own state dict
  - At load time: raw vectors are projected with the CURRENT projection weights
  - After training: just reload — the raw cache is always valid, only projection changes

This guarantees embeddings are always consistent with the current model.

Outputs:
  data/task_raw_embeddings.pt   — dict: task_string -> Tensor (312,)  [permanent]
  data/projection_layer_init.pt — Linear(312->256) initial state dict  [do NOT overwrite after training]

Usage:
  # Build raw cache (one-time, before training):
  python embeddings.py --model-dir tinybert_local --out-dir data

  # In pipeline / train.py (before training — use init weights):
  from embeddings import load_projected_embeddings, get_embedding, TaskProjection
  cache, projection = load_projected_embeddings(
      "data/task_raw_embeddings.pt",
      "data/projection_layer_init.pt"       # untrained, before train.py
  )

  # After training — use trained weights:
  cache, projection = load_projected_embeddings(
      "data/task_raw_embeddings.pt",
      "data/projection_layer_trained.pt"    # saved by train.py
  )
  t = get_embedding("serve wine", cache)   # Tensor (256,)
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_definitions import PAPER_TASKS, SRS_TASKS, NUM_TASKS

TINYBERT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
TINYBERT_DIM   = 312
WORKING_DIM    = 256


# ─────────────────────────────────────────────────────────────────────────────
# Projection layer
# ─────────────────────────────────────────────────────────────────────────────

class TaskProjection(nn.Module):
    """
    Linear(312 -> 256). Trained jointly with TCFG/AGCA/SCRN. TinyBERT frozen.
    Xavier-uniform initialised for stable early gradients.

    Input : Tensor (..., 312)
    Output: Tensor (..., 256)   — NOT L2-normalised here (done in pipeline.py)
    """
    def __init__(self, in_dim: int = TINYBERT_DIM, out_dim: int = WORKING_DIM):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# TinyBERT loading and encoding
# ─────────────────────────────────────────────────────────────────────────────

def load_tinybert(model_name_or_path: str, device: str = "cpu"):
    """
    Load frozen TinyBERT tokenizer + model.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local directory path.
    device : str

    Returns
    -------
    tokenizer, model  (model is frozen, eval mode)
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading TinyBERT from '{model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model     = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. {n_params:,} parameters (all frozen).")
    return tokenizer, model


@torch.inference_mode()
def encode_text(text: str, tokenizer, model, device: str = "cpu") -> torch.Tensor:
    """
    Encode text to 312-D CLS token using TinyBERT.

    Returns Tensor (312,) float32 on CPU.
    """
    inputs  = tokenizer(text, return_tensors="pt", padding=True,
                        truncation=True, max_length=64).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # (312,)


# ─────────────────────────────────────────────────────────────────────────────
# Cache construction (raw 312-D)
# ─────────────────────────────────────────────────────────────────────────────

def compute_raw_embeddings(tokenizer, model, device: str = "cpu") -> dict:
    """
    Compute raw 312-D CLS embeddings for ALL paper tasks + ALL SRS tasks.

    Stores both so the pipeline can look up either string type.
    Keys: task query string → Tensor (312,) CPU float32.

    Note: SRS task strings are also cached so get_embedding() works with
    contest evaluation queries directly.
    """
    all_queries = set(PAPER_TASKS.values()) | set(SRS_TASKS.values())
    raw_cache   = {}

    print(f"\n  Encoding {len(all_queries)} unique task queries...")
    for query in sorted(all_queries):
        vec = encode_text(query, tokenizer, model, device)
        if vec.shape != (TINYBERT_DIM,):
            raise ValueError(f"TinyBERT output wrong shape for '{query}': {vec.shape}")
        raw_cache[query] = vec

    print(f"  Done. {len(raw_cache)} embeddings, each Tensor({TINYBERT_DIM},)")
    return raw_cache


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_raw_cache(raw_cache: dict, projection: "TaskProjection", out_dir: str,
                   force: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    raw_path  = os.path.join(out_dir, "task_raw_embeddings.pt")
    proj_path = os.path.join(out_dir, "projection_layer_init.pt")  # init only

    torch.save(raw_cache, raw_path)

    if force or not os.path.exists(proj_path):
        torch.save(projection.state_dict(), proj_path)
        print(f"  ✓ {proj_path}")
    else:
        print(f"  ⚠ {proj_path} already exists — not overwriting "
              f"(pass --force to replace)")
    print(f"  ✓ {raw_path}   ({len(raw_cache)} raw 312-D embeddings)")

def _safe_load(path: str, device: str = "cpu"):
    """torch.load with weights_only=True, with fallback for older PyTorch."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        print(f"[WARN] weights_only not supported — loading {path} with full pickle "
              f"(ensure file is from a trusted source)")
        return torch.load(path, map_location=device)

def load_projected_embeddings(
    raw_path:  str,
    proj_path: str,
    device:    str = "cpu",
) -> tuple["dict[str, torch.Tensor]", "TaskProjection"]:
    """
    Load raw cache + projection, return projected 256-D cache + projection module.

    Called at pipeline startup. After training, call again with updated proj_path
    to get fresh embeddings — no need to re-run TinyBERT.

    Returns
    -------
    projected_cache : dict { query_string -> Tensor (256,) CPU }
    projection      : TaskProjection (loaded weights, on device)
    """
    raw_cache  = _safe_load(raw_path,  "cpu")
    projection = TaskProjection()
    projection.load_state_dict(_safe_load(proj_path, "cpu"))
    projection.to(device).eval()

    projected_cache = {}
    with torch.no_grad():
        for query, raw_vec in raw_cache.items():
            proj_vec = projection(raw_vec.to(device))
            proj_vec = F.normalize(proj_vec.unsqueeze(0), dim=1).squeeze(0).cpu()  # L2-norm (256,)
            projected_cache[query.lower().strip()] = proj_vec

    return projected_cache, projection


# ─────────────────────────────────────────────────────────────────────────────
# get_embedding
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding(
    task_query:  str,
    cache:       dict,
    tokenizer=None,
    model=None,
    projection:  "TaskProjection | None" = None,
    device:      str  = "cpu",
    allow_novel: bool = False,
) -> torch.Tensor:
    """
    Return 256-D task embedding for a query string.

    Exact match only against the cache (no substring matching — unsafe).
    Novel-prompt fallback requires allow_novel=True (off during contest eval).

    Parameters
    ----------
    task_query  : str  — must exactly match a key in cache for fast path
    cache       : dict — projected_cache from load_projected_embeddings()
    tokenizer   : required only if allow_novel=True
    model       : required only if allow_novel=True
    projection  : required only if allow_novel=True
    device      : str
    allow_novel : bool — disabled during contest evaluation (SRS FR-04)

    Returns
    -------
    Tensor (256,) float32
    """
    # Exact match (O(1))
    if task_query.lower().strip() in cache:   # new
        return cache[task_query.lower().strip()]

    if allow_novel:
        if tokenizer is None or model is None or projection is None:
            raise ValueError(
                "tokenizer, model, projection are required for novel task queries."
            )
        print(f"  [FALLBACK] Novel query '{task_query}' — running TinyBERT on-the-fly "
              f"(~120ms — exceeds 80ms SLA, contest eval only uses cached queries)")
        raw_vec = encode_text(task_query, tokenizer, model, device)
        with torch.no_grad():
            return projection(raw_vec.to(device)).cpu()

    known = sorted(cache.keys())
    raise KeyError(
        f"Unknown task query: '{task_query}'.\n"
        f"Known queries ({len(known)}): {known}\n"
        f"Pass allow_novel=True to enable on-the-fly TinyBERT fallback."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build TADS-X task embedding cache using TinyBERT."
    )
    parser.add_argument("--force", action="store_true",
                    help="Overwrite existing projection_layer_init.pt if present.")
    parser.add_argument(
        "--out-dir", type=str, default="data",
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for TinyBERT inference (default: cpu)"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help=(
            "Local TinyBERT model directory. If omitted, downloads from HuggingFace.\n"
            "Pre-download: python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('huawei-noah/TinyBERT_General_4L_312D', "
            "local_dir='tinybert_local')\""
        )
    )
    args = parser.parse_args()

    model_source = args.model_dir or TINYBERT_MODEL

    print(f"\n{'='*60}")
    print(f"  TADS-X Task Embedding Builder")
    print(f"  Model  : {model_source}")
    print(f"  Device : {args.device}")
    print(f"  Output : {args.out_dir}/")
    print(f"{'='*60}")

    # Step 1: Load TinyBERT
    tokenizer, tinybert = load_tinybert(model_source, args.device)

    # Step 2: Fresh projection (untrained — will be updated during training)
    projection = TaskProjection()
    projection.to(args.device)
    n_proj = sum(p.numel() for p in projection.parameters())
    print(f"\n  TaskProjection: Linear({TINYBERT_DIM}→{WORKING_DIM}), "
          f"{n_proj:,} trainable params")

    # Step 3: Compute raw 312-D embeddings (paper tasks + SRS tasks)
    raw_cache = compute_raw_embeddings(tokenizer, tinybert, args.device)

    # Step 4: Save raw cache + initial projection
    print(f"\n  Saving...")
    save_raw_cache(raw_cache, projection, args.out_dir, force=args.force)

    # Step 5: Smoke-test the full load → project → get_embedding round-trip
    print(f"\n  Smoke test: load → project → get_embedding...")
    raw_path  = os.path.join(args.out_dir, "task_raw_embeddings.pt")
    proj_path = os.path.join(args.out_dir, "projection_layer_init.pt")
    proj_cache, proj2 = load_projected_embeddings(raw_path, proj_path)

    # Test all paper tasks
    for task_id, query in PAPER_TASKS.items():
        emb = get_embedding(query, proj_cache)
        if emb.shape != (WORKING_DIM,):
            raise ValueError(f"Wrong shape for task {task_id}: {emb.shape}")
    # Test all SRS tasks
    for task_id, query in SRS_TASKS.items():
        emb = get_embedding(query, proj_cache)
        if emb.shape != (WORKING_DIM,):
            raise ValueError(f"Wrong shape for task {task_id}: {emb.shape}")
    print(f"  All {NUM_TASKS} paper + {NUM_TASKS} SRS embeddings: "
          f"shape ({WORKING_DIM},) ✓")

    # Step 6: Pairwise similarity between paper task embeddings
    paper_queries = [PAPER_TASKS[i] for i in range(1, NUM_TASKS + 1)]
    embs = torch.stack([proj_cache[q.lower().strip()] for q in paper_queries])  # (14, 256)
    embs_norm = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
    sim       = embs_norm @ embs_norm.T                                 # (14, 14)

    pairs = []
    for i in range(NUM_TASKS):
        for j in range(i + 1, NUM_TASKS):
            pairs.append((float(sim[i, j]), paper_queries[i], paper_queries[j]))
    pairs.sort(reverse=True)

    print(f"\n  Top 3 most similar paper task pairs (before training):")
    for score, q1, q2 in pairs[:3]:
        print(f"    {score:.4f}  '{q1}' <-> '{q2}'")
    print(f"  Bottom 3 least similar:")
    for score, q1, q2 in pairs[-3:]:
        print(f"    {score:.4f}  '{q1}' <-> '{q2}'")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  After training, reload with:")
    print(f"    from embeddings import load_projected_embeddings, get_embedding")
    print(f"    cache, proj = load_projected_embeddings(")
    print(f"        'data/task_raw_embeddings.pt',")
    print(f"        'data/projection_layer_trained.pt'  # <-- your trained weights file")
    print(f"    )")
    print(f"    t = get_embedding('serve wine', cache)  # Tensor (256,)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
