from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from project.Step08_Abundance.phase4_shared import COHORT_CELL_TYPES, COHORT_PRESORTS


LAYER_ORDER = ("layer1_tcell", "layer2_immune", "layer3_nonimmune")
LAYER_REFERENCES = {
    "layer1_tcell": "CD4 Tnaive",
    "layer2_immune": None,
    "layer3_nonimmune": None,
}
SOURCE_COLUMNS = (
    "orig.ident", "Patient", "disease", "tissue-type", "presorted",
    "disease_group", "Subset_Identity",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _robust_sigma(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    q25, q75 = numeric.quantile([0.25, 0.75])
    return float((q75 - q25) / 1.349)


def _dm_concentration(mean: pd.Series, variance: pd.Series, median_depth: float) -> pd.Series:
    denominator = mean * (1.0 - mean)
    ratio = variance / denominator.where(denominator > 0)
    valid = ratio.gt(1.0 / median_depth) & ratio.lt(1.0)
    result = pd.Series(np.nan, index=mean.index, dtype=float)
    result.loc[valid] = (
        median_depth * (1.0 - ratio.loc[valid])
        / (ratio.loc[valid] * median_depth - 1.0)
    )
    return result.where(result.gt(0))


def _relative_effects(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    factor: str,
    group_1: str,
    group_2: str,
    reference: str,
) -> pd.Series:
    if factor not in metadata or {group_1, group_2} - set(metadata[factor].astype(str)):
        return pd.Series(dtype=float)
    wide = counts.pivot(index="sample_id", columns="cell_type", values="count").fillna(0.0)
    joined = metadata.set_index("sample_id")[[factor]].join(wide, how="inner")
    if reference not in wide.columns:
        return pd.Series(dtype=float)
    log_ratio = np.log(joined[wide.columns].add(0.5)).subtract(
        np.log(joined[reference].add(0.5)), axis=0
    )
    levels = joined[factor].astype(str)
    return (
        log_ratio.loc[levels.eq(group_1)].mean()
        - log_ratio.loc[levels.eq(group_2)].mean()
    )


def _estimate_layer(
    layer_id: str,
    counts_observed: pd.DataFrame,
    metadata: pd.DataFrame,
    configured_cell_types: tuple[str, ...],
    allowed_presorts: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    observed = [ct for ct in configured_cell_types if ct in set(counts_observed["cell_type"])]
    if len(observed) < 3:
        raise ValueError(f"{layer_id} has fewer than three observed configured cell types.")
    samples = metadata.drop_duplicates("sample_id").copy()
    if samples["sample_id"].duplicated().any():
        raise ValueError(f"{layer_id} sample metadata is not sample-constant.")
    complete_index = pd.MultiIndex.from_product(
        [samples["sample_id"].astype(str), observed], names=["sample_id", "cell_type"]
    )
    complete = (
        counts_observed.groupby(["sample_id", "cell_type"], observed=True)["count"].sum()
        .reindex(complete_index, fill_value=0).rename("count").reset_index()
    )
    totals = complete.groupby("sample_id")["count"].sum().rename("total_count")
    if totals.le(0).any():
        raise ValueError(f"{layer_id} contains samples with zero total counts.")
    complete = complete.merge(totals, on="sample_id", how="left", validate="many_to_one")
    complete["proportion"] = complete["count"] / complete["total_count"]
    samples = samples.merge(totals.reset_index(), on="sample_id", how="left", validate="one_to_one")

    matrix = complete.pivot(index="sample_id", columns="cell_type", values="proportion")
    count_matrix = complete.pivot(index="sample_id", columns="cell_type", values="count")
    mean = matrix.mean()
    variance = matrix.var(ddof=1)
    zero = count_matrix.eq(0).mean()
    median_depth = float(totals.median())
    concentration_by_cell = _dm_concentration(mean, variance, median_depth)

    baseline_mask = samples["disease"].astype(str).eq("HC") & samples["tissue"].astype(str).eq("nif")
    baseline_samples = samples.loc[baseline_mask, "sample_id"].astype(str)
    baseline_source = "HC_nif"
    if len(baseline_samples) < 4:
        baseline_samples = samples["sample_id"].astype(str)
        baseline_source = "all_samples_fallback"
    baseline = count_matrix.loc[count_matrix.index.intersection(baseline_samples)].sum().add(0.5)
    baseline = baseline / baseline.sum()
    floor = min(0.005, 0.25 / len(baseline))
    simulation_baseline = baseline.clip(lower=floor)
    simulation_baseline = simulation_baseline / simulation_baseline.sum()

    log_counts = np.log(count_matrix.add(0.5))
    clr = log_counts.subtract(log_counts.mean(axis=1), axis=0)
    donor_labels = samples.set_index("sample_id").loc[clr.index, "donor_id"].astype(str)
    donor_means = clr.groupby(donor_labels).mean()
    donor_noise = float(donor_means.std(ddof=1).median()) if len(donor_means) > 1 else 0.0
    within = clr.groupby(donor_labels).std(ddof=1)
    within_values = within.stack().dropna()
    sample_noise = float(within_values.median()) if len(within_values) else 0.0
    sample_heterogeneity = float(
        np.sqrt(clr.subtract(clr.median()).pow(2).mean(axis=1)).median()
    )
    presort_labels = samples.set_index("sample_id").loc[clr.index, "presort"].astype(str)
    presort_means = clr.groupby(presort_labels).mean()
    batch_proxy = float(presort_means.std(ddof=1).median()) if len(presort_means) > 1 else 0.0

    reference = LAYER_REFERENCES[layer_id]
    if reference not in observed:
        reference = str(baseline.sort_values(ascending=False).index[0])
    disease_effects = _relative_effects(
        complete, samples,
        factor="disease_group", group_1="CD_if", group_2="HC_normal", reference=reference,
    )
    tissue_effects = _relative_effects(
        complete, samples,
        factor="tissue", group_1="if", group_2="nif", reference=reference,
    )
    disease_abs = disease_effects.drop(labels=[reference], errors="ignore").abs().dropna()
    tissue_abs = tissue_effects.drop(labels=[reference], errors="ignore").abs().dropna()
    raw_disease = float(disease_abs.quantile(0.75)) if len(disease_abs) else 0.0
    raw_tissue = float(tissue_abs.quantile(0.75)) if len(tissue_abs) else 0.0
    affected_fraction = float(disease_abs.gt(0.1).mean()) if len(disease_abs) else 0.1

    raw_concentration = float(concentration_by_cell.median())
    if not np.isfinite(raw_concentration):
        raw_concentration = 5.0
    simulation_concentration = float(np.clip(raw_concentration, 15.0, 60.0))
    simulation_donor_noise = float(np.clip(donor_noise, 0.10, 0.50))
    simulation_sample_noise = float(np.clip(sample_noise, 0.10, 0.50))
    simulation_depth = float(totals.median())
    simulation_depth_sd = float(np.clip(_robust_sigma(totals), 0.05 * simulation_depth, 0.25 * simulation_depth))
    simulation_min_depth = int(max(50, round(float(totals.quantile(0.10)))))
    simulation_disease = float(np.clip(raw_disease, 0.30, 0.70))
    simulation_tissue = float(np.clip(raw_tissue, 0.20, 0.60))
    simulation_inflamed = float(np.clip(affected_fraction, 0.10, 0.30))

    parameter_rows = [
        ("n_donors", samples["donor_id"].nunique(), samples["donor_id"].nunique(), "structural count"),
        ("n_samples", samples["sample_id"].nunique(), samples["sample_id"].nunique(), "structural count"),
        ("n_celltypes", len(observed), len(observed), "configured stratum universe"),
        ("baseline_alpha_scale", raw_concentration, simulation_concentration, "clipped to tolerant DM concentration [15,60]"),
        ("total_count_mean", float(totals.mean()), simulation_depth, "median protects against depth outliers"),
        ("total_count_sd", float(totals.std(ddof=1)), simulation_depth_sd, "robust IQR sigma clipped to 5%-25% of median"),
        ("min_count", float(totals.min()), simulation_min_depth, "10th percentile with floor 50"),
        ("donor_noise_sd", donor_noise, simulation_donor_noise, "robust donor-mean CLR SD clipped to [0.10,0.50]"),
        ("sample_noise_sd", sample_noise, simulation_sample_noise, "within-donor CLR SD clipped to [0.10,0.50]; diagnostic for DM"),
        ("sample_level_heterogeneity", sample_heterogeneity, sample_heterogeneity, "reported, not directly injected"),
        ("batch_presort_proxy", batch_proxy, min(batch_proxy, 0.30), "presort CLR-mean SD capped for heterogeneous scenario only"),
        ("zero_frequency", float(count_matrix.eq(0).to_numpy().mean()), float(count_matrix.eq(0).to_numpy().mean()), "reported; induced through depth/concentration"),
        ("low_abundance_q10", float(mean.quantile(0.10)), max(float(mean.quantile(0.10)), floor), "baseline floor protection"),
        ("disease_effect_size", raw_disease, simulation_disease, "75th percentile reference-log-ratio effect clipped to [0.30,0.70]"),
        ("tissue_effect_size", raw_tissue, simulation_tissue, "75th percentile reference-log-ratio effect clipped to [0.20,0.60]"),
        ("inflamed_cell_frac", affected_fraction, simulation_inflamed, "fraction |disease effect|>0.1 clipped to [0.10,0.30]"),
    ]
    parameters = pd.DataFrame(
        parameter_rows,
        columns=["parameter", "raw_estimate", "simulation_value", "adjustment_reason"],
    )
    parameters.insert(0, "layer_id", layer_id)

    cell_summary = pd.DataFrame({
        "layer_id": layer_id,
        "cell_type": mean.index,
        "mean_abundance": mean.values,
        "between_sample_variance": variance.reindex(mean.index).values,
        "zero_frequency": zero.reindex(mean.index).values,
        "estimated_dm_concentration": concentration_by_cell.reindex(mean.index).values,
        "baseline_composition_raw": baseline.reindex(mean.index).values,
        "baseline_composition_simulation": simulation_baseline.reindex(mean.index).values,
        "disease_reference_logratio_effect": disease_effects.reindex(mean.index).values,
        "tissue_reference_logratio_effect": tissue_effects.reindex(mean.index).values,
    })
    included_cell_types = pd.DataFrame({
        "layer_id": layer_id,
        "cell_type": configured_cell_types,
        "inclusion_status": ["included" if ct in observed else "absent" for ct in configured_cell_types],
        "reference_cell_type": [ct == reference for ct in configured_cell_types],
    })
    samples.insert(0, "layer_id", layer_id)
    diagnostics = pd.DataFrame([
        {
            "layer_id": layer_id,
            "check": "configured_presort_match",
            "value": ";".join(sorted(samples["presort"].astype(str).unique())),
            "expected": ";".join(allowed_presorts),
            "status": "passed" if set(samples["presort"].astype(str)).issubset(set(allowed_presorts)) else "failed",
        },
        {"layer_id": layer_id, "check": "minimum_samples", "value": len(samples), "expected": ">=4", "status": "passed" if len(samples) >= 4 else "failed"},
        {"layer_id": layer_id, "check": "minimum_celltypes", "value": len(observed), "expected": ">=3", "status": "passed" if len(observed) >= 3 else "failed"},
        {"layer_id": layer_id, "check": "finite_simulation_parameters", "value": bool(pd.to_numeric(parameters["simulation_value"], errors="coerce").notna().all()), "expected": "true", "status": "passed" if pd.to_numeric(parameters["simulation_value"], errors="coerce").notna().all() else "failed"},
    ])
    proposal = {
        "layer_id": layer_id,
        "reference_cell_type": reference,
        "cell_type_names": list(mean.index),
        "baseline_composition": {str(key): float(value) for key, value in simulation_baseline.items()},
        "baseline_alpha_scale": simulation_concentration,
        "total_count_mean": simulation_depth,
        "total_count_sd": simulation_depth_sd,
        "min_count": simulation_min_depth,
        "donor_noise_sd": simulation_donor_noise,
        "sample_noise_sd_diagnostic": simulation_sample_noise,
        "batch_presort_proxy": min(batch_proxy, 0.30),
        "moderate_disease_effect_size": simulation_disease,
        "weak_disease_effect_size": simulation_disease * 0.5,
        "strong_disease_effect_size": min(1.0, simulation_disease * 1.5),
        "tissue_effect_size": simulation_tissue,
        "interaction_effect_size": simulation_disease * 0.5,
        "inflamed_cell_frac": simulation_inflamed,
        "baseline_source": baseline_source,
    }
    return samples, included_cell_types, parameters, diagnostics, {
        "cell_summary": cell_summary,
        "proposal": proposal,
        "complete_counts": complete,
    }


def estimate_stratified_parameters(
    source_csv: str | Path,
    output_root: str | Path,
    *,
    chunksize: int = 150_000,
) -> Path:
    source = Path(source_csv).resolve()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    grouped_chunks: dict[str, list[pd.DataFrame]] = {layer: [] for layer in LAYER_ORDER}
    for chunk in pd.read_csv(source, usecols=list(SOURCE_COLUMNS), chunksize=chunksize):
        chunk = chunk.rename(columns={
            "orig.ident": "sample_id",
            "Patient": "donor_id",
            "tissue-type": "tissue",
            "presorted": "presort",
            "Subset_Identity": "cell_type",
        })
        chunk["presort"] = chunk["presort"].replace({"CD45+CD3+": "CD3+CD19-"})
        for layer in LAYER_ORDER:
            selected = chunk.loc[
                chunk["presort"].astype(str).isin(COHORT_PRESORTS[layer])
                & chunk["cell_type"].astype(str).isin(COHORT_CELL_TYPES[layer])
            ]
            if selected.empty:
                continue
            grouped = (
                selected.groupby(
                    ["sample_id", "donor_id", "disease", "tissue", "presort", "disease_group", "cell_type"],
                    dropna=False, observed=True,
                ).size().rename("count").reset_index()
            )
            grouped_chunks[layer].append(grouped)

    source_record = {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
        "read_columns": list(SOURCE_COLUMNS),
        "created_at": _utc_now(),
    }
    proposals: dict[str, Any] = {}
    comparison_rows: list[pd.DataFrame] = []
    for layer in LAYER_ORDER:
        layer_root = root / "parameter_estimation" / layer
        layer_root.mkdir(parents=True)
        combined = pd.concat(grouped_chunks[layer], ignore_index=True)
        group_columns = [
            "sample_id", "donor_id", "disease", "tissue", "presort", "disease_group", "cell_type",
        ]
        combined = combined.groupby(group_columns, dropna=False, observed=True)["count"].sum().reset_index()
        metadata = combined[group_columns[:-1]].drop_duplicates()
        samples, cell_types, parameters, diagnostics, extras = _estimate_layer(
            layer, combined[["sample_id", "cell_type", "count"]], metadata,
            COHORT_CELL_TYPES[layer], COHORT_PRESORTS[layer],
        )
        samples.to_csv(layer_root / "included_samples.csv", index=False)
        cell_types.to_csv(layer_root / "included_cell_types.csv", index=False)
        parameters.to_csv(layer_root / "estimated_parameters.csv", index=False)
        diagnostics.to_csv(layer_root / "parameter_diagnostics.csv", index=False)
        extras["cell_summary"].to_csv(layer_root / "summary_statistics.csv", index=False)
        extras["complete_counts"].to_csv(layer_root / "stratified_count_input.csv", index=False)
        estimation_config = {
            "layer_id": layer,
            "allowed_presorts": list(COHORT_PRESORTS[layer]),
            "configured_cell_types": list(COHORT_CELL_TYPES[layer]),
            "baseline_subset": "disease=HC and tissue=nif; all-sample fallback only if <4 samples",
            "zero_completion": True,
            "parameter_tolerance_policy": "robust center/quantile with explicit clipping",
        }
        (layer_root / "estimation_config.yaml").write_text(
            yaml.safe_dump(estimation_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        _write_json(layer_root / "input_manifest.json", {
            **source_record,
            "layer_id": layer,
            "allowed_presorts": list(COHORT_PRESORTS[layer]),
            "configured_cell_types": list(COHORT_CELL_TYPES[layer]),
            "number_included_samples": int(len(samples)),
            "number_included_cell_types": int(cell_types["inclusion_status"].eq("included").sum()),
        })
        proposals[layer] = extras["proposal"]
        comparison_rows.append(parameters)

    comparison = pd.concat(comparison_rows, ignore_index=True)
    comparison.to_csv(root / "layer_parameter_comparison.csv", index=False)
    proposal_document = {
        "schema_version": "phase6-stratified-simulation-parameters-v1",
        "source_manifest": source_record,
        "primary_layer": "layer1_tcell",
        "layers": proposals,
    }
    (root / "simulation_parameter_proposal.yaml").write_text(
        yaml.safe_dump(proposal_document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _write_json(root / "calibration_manifest.json", {
        "schema_version": "phase6-stratified-calibration-v1",
        "status": "complete",
        "created_at": _utc_now(),
        "source": source_record,
        "layers": list(LAYER_ORDER),
    })
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate Phase-6 presort-stratified simulation parameters")
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--chunksize", type=int, default=150_000)
    args = parser.parse_args(argv)
    print(estimate_stratified_parameters(args.source_csv, args.output_root, chunksize=args.chunksize))


if __name__ == "__main__":
    main()
