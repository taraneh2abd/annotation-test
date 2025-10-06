# retrieval.py
import os
import random
from typing import List

def top_k_similar(
    query_path: str,
    all_paths: List[str],
    k: int = 24,
    exclude_self: bool = True
) -> List[str]:
    """
    For testing only:
    Returns k random image paths from all_paths (optionally excluding query).
    """
    if not all_paths:
        return []

    # Normalize and filter
    all_paths = list(dict.fromkeys(all_paths))  # de-duplicate
    candidates = [p for p in all_paths if p != query_path] if exclude_self else all_paths

    if not candidates:
        return []

    k = min(k, len(candidates))
    return random.sample(candidates, k)
