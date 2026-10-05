import numpy as np


def ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    result = np.empty(len(values))
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            result[order[start:end]] = (start + end + 1) / 2
            start = end
    return result


def agreement(ours: np.ndarray, reference: np.ndarray, top: float = 0.1) -> dict:
    """How closely two scorers agree on the same documents: rank and linear correlation, and the share of the
    reference's top fraction that ours also puts in its top fraction (DCLM keeps the top 10%)."""
    k = max(1, int(len(ours) * top))
    top_ours, top_reference = set(np.argsort(-ours)[:k]), set(np.argsort(-reference)[:k])
    return {
        "documents": len(ours),
        "spearman": float(np.corrcoef(ranks(ours), ranks(reference))[0, 1]),
        "pearson": float(np.corrcoef(ours, reference)[0, 1]),
        f"top_{top:g}_overlap": len(top_ours & top_reference) / k,
    }


def spread(paths: list[str], count: int) -> list[str]:
    """`count` paths evenly spaced through the sorted list, so a sample covers the whole run, not its first files."""
    paths = sorted(paths)
    step = max(1, len(paths) // count)
    return paths[::step][:count]
