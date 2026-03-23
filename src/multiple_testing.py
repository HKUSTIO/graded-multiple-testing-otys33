from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    m = len(p_values)
    return p_values <= alpha / m


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    rejections = np.zeros(m, dtype=bool)
    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_pvals[k] <= threshold:
            rejections[sorted_indices[k]] = True
        else:
            break
    return rejections


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    satisfied = sorted_pvals <= thresholds
    rejections = np.zeros(m, dtype=bool)
    if satisfied.any():
        max_k = int(np.max(np.where(satisfied)))
        rejections[sorted_indices[: max_k + 1]] = True
    return rejections


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    m = len(p_values)
    c = float(np.sum(1.0 / np.arange(1, m + 1)))
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    thresholds = (np.arange(1, m + 1) / m) * (alpha / c)
    satisfied = sorted_pvals <= thresholds
    rejections = np.zeros(m, dtype=bool)
    if satisfied.any():
        max_k = int(np.max(np.where(satisfied)))
        rejections[sorted_indices[: max_k + 1]] = True
    return rejections


def compute_fwer(rejections_null: np.ndarray) -> float:
    return float(np.mean(rejections_null.any(axis=1)))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    total_rejections = int(np.sum(rejections))
    if total_rejections == 0:
        return 0.0
    false_discoveries = int(np.sum(rejections & is_true_null))
    return false_discoveries / total_rejections


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    false_nulls = ~is_true_null
    total_false_nulls = int(np.sum(false_nulls))
    if total_false_nulls == 0:
        return 0.0
    true_rejections = int(np.sum(rejections & false_nulls))
    return true_rejections / total_false_nulls


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    null_sim_ids = null_pvalues["sim_id"].unique()
    l_null = len(null_sim_ids)
    m = null_pvalues["hypothesis_id"].nunique()

    rej_null_uncorrected = np.zeros((l_null, m), dtype=bool)
    rej_null_bonferroni = np.zeros((l_null, m), dtype=bool)
    rej_null_holm = np.zeros((l_null, m), dtype=bool)

    for i, sid in enumerate(sorted(null_sim_ids)):
        pvals = null_pvalues[null_pvalues["sim_id"] == sid].sort_values("hypothesis_id")["p_value"].to_numpy()
        rej_null_uncorrected[i] = pvals <= alpha
        rej_null_bonferroni[i] = bonferroni_rejections(pvals, alpha)
        rej_null_holm[i] = holm_rejections(pvals, alpha)

    fwer_uncorrected = compute_fwer(rej_null_uncorrected)
    fwer_bonferroni = compute_fwer(rej_null_bonferroni)
    fwer_holm = compute_fwer(rej_null_holm)

    mixed_sim_ids = mixed_pvalues["sim_id"].unique()

    fdr_list_uncorrected, fdr_list_bh, fdr_list_by = [], [], []
    power_list_uncorrected, power_list_bh, power_list_by = [], [], []

    for sid in sorted(mixed_sim_ids):
        sim_df = mixed_pvalues[mixed_pvalues["sim_id"] == sid].sort_values("hypothesis_id")
        pvals = sim_df["p_value"].to_numpy()
        is_true_null = sim_df["is_true_null"].to_numpy().astype(bool)

        rej_unc = pvals <= alpha
        rej_bh = benjamini_hochberg_rejections(pvals, alpha)
        rej_by = benjamini_yekutieli_rejections(pvals, alpha)

        fdr_list_uncorrected.append(compute_fdr(rej_unc, is_true_null))
        fdr_list_bh.append(compute_fdr(rej_bh, is_true_null))
        fdr_list_by.append(compute_fdr(rej_by, is_true_null))

        power_list_uncorrected.append(compute_power(rej_unc, is_true_null))
        power_list_bh.append(compute_power(rej_bh, is_true_null))
        power_list_by.append(compute_power(rej_by, is_true_null))

    return {
        "fwer_uncorrected": fwer_uncorrected,
        "fwer_bonferroni": fwer_bonferroni,
        "fwer_holm": fwer_holm,
        "fdr_uncorrected": float(np.mean(fdr_list_uncorrected)),
        "fdr_bh": float(np.mean(fdr_list_bh)),
        "fdr_by": float(np.mean(fdr_list_by)),
        "power_uncorrected": float(np.mean(power_list_uncorrected)),
        "power_bh": float(np.mean(power_list_bh)),
        "power_by": float(np.mean(power_list_by)),
    }
