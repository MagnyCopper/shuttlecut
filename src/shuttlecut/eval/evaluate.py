from dataclasses import dataclass, field


@dataclass
class EvalReport:
    recall: float
    precision: float
    mae_s: float
    n_gt: int
    n_det: int
    matched: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    missed: list[tuple[float, float]] = field(default_factory=list)
    extra: list[tuple[float, float]] = field(default_factory=list)
    fragment: list[tuple[float, float]] = field(default_factory=list)
    boundary: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)


def _inter(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def evaluate(detected: list[tuple[float, float]], gt: list[tuple[float, float]],
             tol_s: float = 2.5, overlap: float = 0.5,
             mae_report_s: float = 1.5) -> EvalReport:
    used: set[int] = set()
    matches: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for g in gt:
        best, best_i = 0.0, -1
        for i, d in enumerate(detected):
            if i in used:
                continue
            ov = _inter(d, g)
            if ov <= 0:
                continue
            ratio = ov / max(d[1] - d[0], g[1] - g[0])
            ds, de = abs(d[0] - g[0]), abs(d[1] - g[1])
            if ratio >= overlap and ds <= tol_s and de <= tol_s and ov > best:
                best, best_i = ov, i
        if best_i >= 0:
            used.add(best_i)
            d = detected[best_i]
            matches.append((d, g, 0.5 * (abs(d[0] - g[0]) + abs(d[1] - g[1]))))
    matched_gt = {id(g) for _, g, _ in matches}
    missed = [g for g in gt if id(g) not in matched_gt]
    unmatched = [d for i, d in enumerate(detected) if i not in used]
    extra, fragment = [], []
    for d in unmatched:
        overlapped = any(_inter(d, g) > 0 for _, g, _ in matches)
        (fragment if overlapped else extra).append(d)
    mae = sum(m for _, _, m in matches) / len(matches) if matches else float("nan")
    boundary = [(d, g) for d, g, m in matches if m > mae_report_s]
    return EvalReport(
        recall=len(matches) / len(gt) if gt else 0.0,
        precision=len(matches) / len(detected) if detected else 0.0,
        mae_s=mae, n_gt=len(gt), n_det=len(detected),
        matched=[(d, g) for d, g, _ in matches], missed=missed,
        extra=extra, fragment=fragment, boundary=boundary,
    )
