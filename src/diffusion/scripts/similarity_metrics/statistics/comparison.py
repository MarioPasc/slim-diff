"""Statistical comparison functions for similarity metrics.

Provides non-parametric tests for comparing metrics across experiments:
- Within-group: Friedman + Nemenyi (compare Lp norms within prediction types)
- Between-group: Kruskal-Wallis + Dunn's (compare prediction types)
- Effect sizes: Cliff's delta
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Try to import scikit-posthocs for Nemenyi and Dunn's tests
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("Warning: scikit-posthocs not installed. Post-hoc tests will use fallback.")


def compute_cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta effect size.

    Cliff's delta measures the degree of overlap between two groups.
    Range: [-1, 1], where:
        - 1: all values in group1 > all values in group2
        - 0: groups are stochastically equal
        - -1: all values in group1 < all values in group2

    Interpretation (Romano et al., 2006):
        - |d| < 0.147: negligible
        - |d| < 0.33: small
        - |d| < 0.474: medium
        - |d| >= 0.474: large

    Args:
        group1: First group of values.
        group2: Second group of values.

    Returns:
        Cliff's delta value.
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return float("nan")

    # Count pairs where group1 > group2 and group1 < group2
    dominance = 0
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1

    return dominance / (n1 * n2)


def interpret_cliffs_delta(d: float) -> str:
    """Interpret Cliff's delta magnitude.

    Args:
        d: Cliff's delta value.

    Returns:
        Interpretation string.
    """
    if np.isnan(d):
        return "undefined"
    d_abs = abs(d)
    if d_abs < 0.147:
        return "negligible"
    elif d_abs < 0.33:
        return "small"
    elif d_abs < 0.474:
        return "medium"
    else:
        return "large"


def within_group_comparison(
    metrics_df: pd.DataFrame,
    group_col: str = "prediction_type",
    treatment_col: str = "lp_norm",
    value_col: str = "kid_global",
    block_col: str = "replica_id",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare treatments within each group using Friedman test.

    For each prediction type (epsilon, velocity, x0), compares the 3 Lp norm
    values (1.5, 2.0, 2.5) across 5 replicas using non-parametric repeated
    measures ANOVA (Friedman test) with Nemenyi post-hoc if significant.

    Args:
        metrics_df: DataFrame with columns for group, treatment, value, and block.
        group_col: Column defining groups (e.g., "prediction_type").
        treatment_col: Column defining treatments to compare (e.g., "lp_norm").
        value_col: Column with metric values.
        block_col: Column defining blocks/subjects (e.g., "replica_id").
        alpha: Significance level.

    Returns:
        DataFrame with Friedman test results per group:
            - group: Group identifier
            - friedman_stat: Test statistic
            - p_value: Raw p-value
            - significant: Whether p < alpha
            - best_treatment: Treatment with lowest mean value
            - posthoc_comparisons: Dict of pairwise p-values (if significant)
    """
    results = []
    groups = metrics_df[group_col].unique()

    for group in groups:
        group_df = metrics_df[metrics_df[group_col] == group]

        # Pivot to get treatments as columns, blocks as rows
        # Use pivot_table with mean aggregation to handle duplicate entries
        pivot = group_df.pivot_table(
            index=block_col,
            columns=treatment_col,
            values=value_col,
            aggfunc="mean",
        )

        treatments = pivot.columns.tolist()
        n_treatments = len(treatments)
        n_blocks = len(pivot)

        if n_treatments < 2 or n_blocks < 3:
            results.append({
                "group": group,
                "friedman_stat": float("nan"),
                "p_value": float("nan"),
                "significant": False,
                "best_treatment": treatments[0] if treatments else None,
                "posthoc_comparisons": {},
                "treatment_means": {},
            })
            continue

        # Friedman test
        try:
            stat, p_value = stats.friedmanchisquare(*[pivot[t].values for t in treatments])
        except ValueError as e:
            print(f"Warning: Friedman test failed for {group}: {e}")
            stat, p_value = float("nan"), float("nan")

        # Identify best treatment (lowest mean)
        treatment_means = {t: pivot[t].mean() for t in treatments}
        best_treatment = min(treatment_means, key=treatment_means.get)

        # Post-hoc Nemenyi test if significant
        posthoc_comparisons = {}
        if not np.isnan(p_value) and p_value < alpha:
            if HAS_POSTHOCS:
                try:
                    posthoc_df = sp.posthoc_nemenyi_friedman(pivot)
                    # Extract pairwise comparisons
                    for i, t1 in enumerate(treatments):
                        for j, t2 in enumerate(treatments):
                            if i < j:
                                key = f"{t1}_vs_{t2}"
                                posthoc_comparisons[key] = float(posthoc_df.loc[t1, t2])
                except Exception as e:
                    print(f"Warning: Nemenyi post-hoc failed: {e}")
            else:
                # Fallback: pairwise Wilcoxon with Bonferroni
                n_comparisons = n_treatments * (n_treatments - 1) // 2
                for i, t1 in enumerate(treatments):
                    for j, t2 in enumerate(treatments):
                        if i < j:
                            try:
                                _, p_pair = stats.wilcoxon(pivot[t1], pivot[t2])
                                p_pair_corrected = min(p_pair * n_comparisons, 1.0)
                                key = f"{t1}_vs_{t2}"
                                posthoc_comparisons[key] = p_pair_corrected
                            except ValueError:
                                pass

        results.append({
            "group": group,
            "friedman_stat": float(stat) if not np.isnan(stat) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "significant": p_value < alpha if not np.isnan(p_value) else False,
            "best_treatment": best_treatment,
            "posthoc_comparisons": posthoc_comparisons,
            "treatment_means": treatment_means,
        })

    return pd.DataFrame(results)


def between_group_comparison(
    metrics_df: pd.DataFrame,
    group_col: str = "prediction_type",
    value_col: str = "kid_global",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare groups using Kruskal-Wallis test.

    Compares prediction types (epsilon, velocity, x0) using all samples
    (aggregated across Lp norms) with non-parametric one-way ANOVA
    (Kruskal-Wallis H-test) and Dunn's post-hoc with FDR correction.

    Args:
        metrics_df: DataFrame with columns for group and value.
        group_col: Column defining groups to compare.
        value_col: Column with metric values.
        alpha: Significance level.

    Returns:
        Dict with:
            - kruskal_stat: Test statistic
            - p_value: Raw p-value
            - significant: Whether p < alpha
            - best_group: Group with lowest mean value
            - group_stats: Dict of group -> {mean, std, n}
            - posthoc_pvalues: Dict of pairwise comparisons
            - effect_sizes: Dict of pairwise Cliff's delta
    """
    groups = metrics_df[group_col].unique()
    group_values = {g: metrics_df[metrics_df[group_col] == g][value_col].values for g in groups}

    # Basic stats per group
    group_stats = {}
    for g, values in group_values.items():
        group_stats[g] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }

    # Kruskal-Wallis test
    try:
        stat, p_value = stats.kruskal(*group_values.values())
    except ValueError as e:
        print(f"Warning: Kruskal-Wallis test failed: {e}")
        stat, p_value = float("nan"), float("nan")

    # Identify best group (lowest mean)
    best_group = min(group_stats.keys(), key=lambda g: group_stats[g]["mean"])

    # Post-hoc Dunn's test with FDR correction
    posthoc_pvalues = {}
    posthoc_significant = {}
    effect_sizes = {}

    if not np.isnan(p_value) and p_value < alpha:
        if HAS_POSTHOCS:
            try:
                # Prepare data for posthoc test as DataFrame
                all_values = []
                all_groups = []
                for g, values in group_values.items():
                    all_values.extend(values.tolist() if hasattr(values, 'tolist') else list(values))
                    all_groups.extend([g] * len(values))

                # Create DataFrame for posthoc_dunn (required format for scikit-posthocs >= 0.8)
                posthoc_data = pd.DataFrame({
                    'value': all_values,
                    'group': all_groups,
                })
                posthoc_df = sp.posthoc_dunn(
                    posthoc_data,
                    val_col='value',
                    group_col='group',
                    p_adjust="fdr_bh",
                )

                # Extract pairwise comparisons
                groups_list = list(groups)
                for i, g1 in enumerate(groups_list):
                    for j, g2 in enumerate(groups_list):
                        if i < j:
                            key = f"{g1}_vs_{g2}"
                            p_val = float(posthoc_df.loc[g1, g2])
                            posthoc_pvalues[key] = p_val
                            posthoc_significant[key] = p_val < alpha
            except Exception as e:
                print(f"Warning: Dunn's post-hoc failed: {e}")
        else:
            # Fallback: pairwise Mann-Whitney with FDR correction
            pvalues_raw = []
            comparisons = []
            groups_list = list(groups)
            for i, g1 in enumerate(groups_list):
                for j, g2 in enumerate(groups_list):
                    if i < j:
                        try:
                            _, p_pair = stats.mannwhitneyu(
                                group_values[g1],
                                group_values[g2],
                                alternative="two-sided",
                            )
                            comparisons.append(f"{g1}_vs_{g2}")
                            pvalues_raw.append(p_pair)
                        except ValueError:
                            pass

            if pvalues_raw:
                _, pvalues_corrected, _, _ = multipletests(pvalues_raw, method="fdr_bh")
                for comp, p_corr in zip(comparisons, pvalues_corrected):
                    posthoc_pvalues[comp] = float(p_corr)
                    posthoc_significant[comp] = p_corr < alpha

    # Compute effect sizes (Cliff's delta) for all pairs
    groups_list = list(groups)
    for i, g1 in enumerate(groups_list):
        for j, g2 in enumerate(groups_list):
            if i < j:
                key = f"{g1}_vs_{g2}"
                d = compute_cliffs_delta(group_values[g1], group_values[g2])
                effect_sizes[key] = {
                    "cliffs_delta": d,
                    "interpretation": interpret_cliffs_delta(d),
                }

    return {
        "kruskal_stat": float(stat) if not np.isnan(stat) else None,
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "significant": p_value < alpha if not np.isnan(p_value) else False,
        "best_group": best_group,
        "group_stats": group_stats,
        "posthoc_pvalues": posthoc_pvalues,
        "posthoc_significant": posthoc_significant,
        "effect_sizes": effect_sizes,
    }


def run_all_comparisons(
    metrics_df: pd.DataFrame,
    metrics: list[str] = ["kid_global", "fid_global", "lpips_global"],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run all statistical comparisons for multiple metrics.

    Args:
        metrics_df: DataFrame with metric columns.
        metrics: List of metric column names to analyze.
        alpha: Significance level.

    Returns:
        Dict with results for each metric:
            - within_group: DataFrame of Friedman test results
            - between_group: Dict of Kruskal-Wallis results
    """
    results = {}

    for metric in metrics:
        if metric not in metrics_df.columns:
            print(f"Warning: {metric} not found in DataFrame, skipping...")
            continue

        # Within-group comparison (Lp norms within prediction types)
        within_results = within_group_comparison(
            metrics_df,
            group_col="prediction_type",
            treatment_col="lp_norm",
            value_col=metric,
            block_col="replica_id",
            alpha=alpha,
        )

        # Between-group comparison (prediction types)
        between_results = between_group_comparison(
            metrics_df,
            group_col="prediction_type",
            value_col=metric,
            alpha=alpha,
        )

        results[metric] = {
            "within_group": within_results,
            "between_group": between_results,
        }

    return results


def comparison_results_to_dataframe(
    comparison_results: dict[str, Any],
) -> pd.DataFrame:
    """Convert comparison results to a flat DataFrame for CSV export.

    Args:
        comparison_results: Output from run_all_comparisons().

    Returns:
        DataFrame with one row per comparison.
    """
    rows = []

    for metric, results in comparison_results.items():
        # Within-group results
        for _, row in results["within_group"].iterrows():
            rows.append({
                "metric": metric,
                "comparison_type": "within_group",
                "group": row["group"],
                "test_name": "friedman",
                "test_statistic": row["friedman_stat"],
                "p_value": row["p_value"],
                "significant": row["significant"],
                "best_config": f"{row['group']}_lp_{row['best_treatment']}",
            })

        # Between-group results
        bg = results["between_group"]
        rows.append({
            "metric": metric,
            "comparison_type": "between_group",
            "group": "all",
            "test_name": "kruskal_wallis",
            "test_statistic": bg["kruskal_stat"],
            "p_value": bg["p_value"],
            "significant": bg["significant"],
            "best_config": bg["best_group"],
        })

        # Post-hoc pairwise comparisons
        for comp_key, p_val in bg.get("posthoc_pvalues", {}).items():
            effect = bg.get("effect_sizes", {}).get(comp_key, {})
            rows.append({
                "metric": metric,
                "comparison_type": "posthoc",
                "group": comp_key,
                "test_name": "dunn",
                "test_statistic": None,
                "p_value": p_val,
                "significant": bg.get("posthoc_significant", {}).get(comp_key, False),
                "best_config": None,
                "effect_size": effect.get("cliffs_delta"),
                "effect_interpretation": effect.get("interpretation"),
            })

    return pd.DataFrame(rows)


def get_significance_code(p_value: float | None) -> str:
    """Get significance code for p-value.

    Args:
        p_value: P-value (can be None or NaN).

    Returns:
        Significance code: '', '*', '**', '***'.
    """
    if p_value is None or np.isnan(p_value):
        return ""
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""
