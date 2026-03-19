import re
from typing import Any, Dict, List, Optional, Tuple


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    result = [list(sorted(intervals)[0])]
    for s, e in sorted(intervals)[1:]:
        if s <= result[-1][1]:
            result[-1][1] = max(result[-1][1], e)
        else:
            result.append([s, e])
    return [tuple(iv) for iv in result]


def union_size(intervals: List[Tuple[int, int]]) -> int:
    return sum(e - s for s, e in merge_intervals(intervals))


def intersection_size(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> int:
    ma, mb = merge_intervals(a), merge_intervals(b)
    i, j, total = 0, 0, 0
    while i < len(ma) and j < len(mb):
        lo = max(ma[i][0], mb[j][0])
        hi = min(ma[i][1], mb[j][1])
        if lo < hi:
            total += hi - lo
        if ma[i][1] < mb[j][1]:
            i += 1
        else:
            j += 1
    return total


def compute_metrics(
    retrieved: List[Tuple[int, int]], evidence: List[Tuple[int, int]]
) -> Dict[str, float]:
    covered = intersection_size(retrieved, evidence)
    total_retrieved = union_size(retrieved)
    total_evidence = union_size(evidence)
    precision = covered / total_retrieved if total_retrieved else 0.0
    recall = covered / total_evidence if total_evidence else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def parse_answer_ranges(answer_str: str) -> List[Tuple[int, int]]:
    return [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r"\[(\d+),\s*(\d+)\]", answer_str)
    ]


def _build_intervals(
    answer_str: str,
    paper_text: str,
    paper_lines: List[Dict[str, Any]],
    ground_truth_evidence: List[str],
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    ranges = parse_answer_ranges(answer_str)
    if not ranges:
        return None

    n = len(paper_lines)
    retrieved_intervals = []
    for start, end in ranges:
        s = max(0, start)
        e = min(n - 1, end)
        if s <= e:
            retrieved_intervals.append((paper_lines[s]["char_start"], paper_lines[e]["char_end"]))

    if not retrieved_intervals:
        return None

    evidence_intervals = []
    for ev in ground_truth_evidence:
        trimmed = ev.strip()
        idx = paper_text.find(trimmed)
        if idx != -1:
            evidence_intervals.append((idx, idx + len(trimmed)))

    if not evidence_intervals:
        return None

    return retrieved_intervals, evidence_intervals


def compute_all_metrics(
    solution_str: str,
    ground_truth_evidence: List[str],
    paper_text: str,
    paper_lines: List[Dict[str, Any]],
) -> Dict[str, float]:
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not answer_match:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    result = _build_intervals(answer_match.group(1), paper_text, paper_lines, ground_truth_evidence)
    if result is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    retrieved_intervals, evidence_intervals = result
    return compute_metrics(retrieved_intervals, evidence_intervals)


def compute_score(
    solution_str: str,
    ground_truth_evidence: List[str],
    paper_text: str,
    paper_lines: List[Dict[str, Any]],
) -> float:
    return compute_all_metrics(solution_str, ground_truth_evidence, paper_text, paper_lines)["f1"]
