import inspect
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..schemes.detector import DetectionResult, WatermarkDetector


def compute_avg_tpr_at_milestones(
    results: list[DetectionResult],
    significance_levels: list[float],
    step_size: int,
) -> dict[float, list[float]]:
    if not results:
        return

    res = results[0]
    tpr_at_milestones = defaultdict(list)

    # for RDF, using `p_value_at_milestones`
    if hasattr(res, "p_value_at_milestones") and res.p_value_at_milestones:
        max_len = max(len(r.p_value_at_milestones) for r in results)
        for i in range(max_len):
            p_values = [
                (
                    r.p_value_at_milestones[i]
                    if len(r.p_value_at_milestones) > i
                    else r.p_value_at_milestones[-1]
                )
                for r in results
                if r.p_value_at_milestones
            ]
            for sl in significance_levels:
                tpr = sum(p < sl for p in p_values) / len(p_values) if p_values else 0.0
                tpr_at_milestones[sl].append(tpr)
        return tpr_at_milestones

    # for KGW, using `z_score_at_T`
    if hasattr(res, "z_score_at_T") and res.z_score_at_T is not None:
        max_len = max(len(r.z_score_at_T) for r in results)
        milestones = range(0, max_len, step_size)
        z_thresholds = [float(stats.norm.ppf(1 - sl)) for sl in significance_levels]
        for i in milestones:
            z_scores = [
                (
                    r.z_score_at_T[i].item()
                    if len(r.z_score_at_T) > i
                    else r.z_score_at_T[-1].item()
                )
                for r in results
                if r.z_score_at_T is not None and len(r.z_score_at_T) > 0
            ]
            for t, sl in zip(z_thresholds, significance_levels):
                tpr = sum(z > t for z in z_scores) / len(z_scores) if z_scores else 0.0
                tpr_at_milestones[sl].append(tpr)
        return tpr_at_milestones

    # for VOW, using `p_value_per_token`
    if hasattr(res, "p_values_per_token") and res.p_values_per_token:
        max_len = max(len(r.p_values_per_token) for r in results)
        milestones = range(0, max_len, step_size)
        for i in milestones:
            p_values = [
                (
                    r.p_values_per_token[i]
                    if len(r.p_values_per_token) > i
                    else r.p_values_per_token[-1]
                )
                for r in results
                if r.p_values_per_token
            ]
            for sl in significance_levels:
                tpr = sum(p < sl for p in p_values) / len(p_values) if p_values else 0.0
                tpr_at_milestones[sl].append(tpr)
        return tpr_at_milestones

    raise ValueError(f"Unknown detection result with keys: {vars(res)}")


def compute_auc_at_milestones(
    results: list[DetectionResult], negative_indices: list[int], step_size: int
) -> list[float]:
    res = results[0]
    auc_at_milestones = []

    # for RDF, using `p_value_at_milestones`
    if hasattr(res, "p_value_at_milestones") and res.p_value_at_milestones:
        max_len = max(len(r.p_value_at_milestones) for r in results)
        for i in range(max_len):
            y_score = []
            y_true = []
            for j, r in enumerate(results):
                if not r.p_value_at_milestones:
                    continue
                if len(r.p_value_at_milestones) > i:
                    score = r.p_value_at_milestones[i]
                else:
                    score = r.p_value_at_milestones[-1]
                label = 2 if j in negative_indices else 1
                y_score.append(score)
                y_true.append(label)
            auc = roc_auc_score(y_true, y_score)
            auc_at_milestones.append(auc)
        return auc_at_milestones

    # for KGW, using `z_score_at_T`
    if hasattr(res, "z_score_at_T") and res.z_score_at_T is not None:
        max_len = max(len(r.z_score_at_T) for r in results)
        for i in range(0, max_len, step_size):
            y_score = []
            y_true = []
            for j, r in enumerate(results):
                if r.z_score_at_T is None or len(r.z_score_at_T) == 0:
                    continue
                if len(r.z_score_at_T) > i:
                    score = r.z_score_at_T[i].item()
                else:
                    score = r.z_score_at_T[-1].item()
                label = 0 if j in negative_indices else 1
                y_score.append(score)
                y_true.append(label)
            auc = roc_auc_score(y_true, y_score)
            auc_at_milestones.append(auc)
        return auc_at_milestones

    # for VOW, using `p_value_per_token`
    if hasattr(res, "p_values_per_token") and res.p_values_per_token:
        max_len = max(len(r.p_values_per_token) for r in results)
        for i in range(0, max_len, step_size):
            y_score = []
            y_true = []
            for j, r in enumerate(results):
                if not r.p_values_per_token:
                    continue
                if len(r.p_values_per_token) > i:
                    score = r.p_values_per_token[i]
                else:
                    score = r.p_values_per_token[-1]
                label = 2 if j in negative_indices else 1
                y_score.append(score)
                y_true.append(label)
            auc = roc_auc_score(y_true, y_score)
            auc_at_milestones.append(auc)
        return auc_at_milestones

    raise ValueError(f"Unknown detection result with keys: {vars(res)}")


def detect_texts(
    detector: WatermarkDetector,
    texts: list[str],
    token_num: int | None = None,
    significance_levels: list[float] | None = None,
    use_local: bool = True,
    step_size: int | None = None,
    negative_indices: list[int] | None = None,
    include_auc: bool = False,
    include_raw_results: bool = False,
    include_raw_costs: bool = False,
    **kwargs,
) -> dict:
    if significance_levels is None:
        significance_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    batch_detect = detector.batch_detect
    if use_local and hasattr(detector, "local_batch_detect"):
        batch_detect = detector.local_batch_detect

    parameters = list(inspect.signature(batch_detect).parameters.keys())

    report_per_token_tpr = step_size is not None and step_size > 0

    if "step_size" in parameters:
        kwargs.update({"step_size": step_size})
    if "include_p_values_per_token" in parameters:
        kwargs.update({"include_p_values_per_token": report_per_token_tpr})
    if "return_z_at_T" in parameters:
        kwargs.update({"return_z_at_T": report_per_token_tpr})

    results = batch_detect(texts, token_num=token_num, **kwargs)
    costs = None
    if isinstance(results, tuple):
        results, costs = results

    final_result = {}
    # keys:
    # - tpr_dict
    # - p_value_median
    # - tpr_per_step
    # - auc
    # - overhead_per_sample
    # - avg_tokens_per_sample
    # - raw_costs
    # - raw_results

    if negative_indices is None:
        negative_indices = []

    # 1. calculate TPRs at fixed significance levels
    tpr_dict = {}

    filtered_results = [r for i, r in enumerate(results) if i not in negative_indices]
    for sl in significance_levels:
        detected_num = sum(r.p_value < sl for r in filtered_results)
        tpr_dict[sl] = detected_num / len(filtered_results)
    final_result["tpr_dict"] = tpr_dict

    # 2. calculate median of p-values
    p_value_array = np.array([r.p_value for r in filtered_results])
    p_value_median = np.median(p_value_array)
    final_result["p_value_median"] = p_value_median

    # 3. calculate TPR per step_size
    tpr_per_step = None
    if report_per_token_tpr:
        tpr_per_step = compute_avg_tpr_at_milestones(
            filtered_results,
            significance_levels,
            step_size,
        )
    final_result["tpr_per_step"] = tpr_per_step

    # 4. calculate AUC if requested
    auc = None
    if include_auc and not negative_indices:
        print("Warning: computing AUC requires negative samples, ignored.")
    elif include_auc:
        y_score = [r.p_value for r in results]
        y_true = [1] * len(results)
        for idx in negative_indices:
            y_true[idx] = 2
        auc = roc_auc_score(y_true, y_score)
    final_result["auc"] = auc

    # 5. calculate overall detection overhead
    overhead_per_sample = (
        float(np.mean([c.total_time for c in costs])) if costs else None
    )
    final_result["overhead_per_sample"] = overhead_per_sample

    # 6. calculate average tokens per sample
    avg_tokens_per_sample = float(np.mean([r.total_token_num for r in filtered_results]))
    final_result["avg_tokens_per_sample"] = avg_tokens_per_sample

    if include_raw_costs:
        final_result["raw_costs"] = costs
    if include_raw_results:
        final_result["raw_results"] = results

    return final_result
