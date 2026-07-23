from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from src.stats.adapters import (
    DCATSAdapter,
    MockBayesianAdapter,
    MockFailureAdapter,
    MockFrequentistAdapter,
    NaiveWelchProportionAdapter,
    PropellerAdapter,
    RScriptBridge,
    ScCODAAdapter,
    SccompAdapter,
)
from src.stats.evaluation import EvaluationResult, EvaluationSpec, evaluate_contrasts
from src.stats.meta_engine.Tri_anchor import TriAnchorAdapter, load_tri_anchor_rule
from src.stats.runners import DifferentialAbundanceRunner
from src.stats.schemas import (
    CanonicalDAInput,
    DecisionRule,
    load_default_decision_rules,
    validate_truth_table,
)
from src.stats.simulation import simulate_DM_data
from src.stats.validation import parse_boolean_series


@dataclass(frozen=True)
class PipelineRunResult:
    run_id: str
    mode: str
    output_root: Path
    public_view: pd.DataFrame
    evidence_layer: pd.DataFrame
    diagnostics: pd.DataFrame
    truth_table: pd.DataFrame | None
    evaluation: EvaluationResult | None
    summary_tables: dict[str, pd.DataFrame]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("Pipeline configuration must be a mapping.")
    return document


def _validate_config(config: dict[str, Any]) -> None:
    required = {"mode", "run_id", "analysis_id", "methods", "output_root", "contrast"}
    if missing := required - set(config):
        raise ValueError(f"Pipeline configuration is missing: {sorted(missing)}")
    if config["mode"] not in {"simulation", "real_data"}:
        raise ValueError("mode must be 'simulation' or 'real_data'.")
    if not config["run_id"] or str(config["run_id"]) == str(config["analysis_id"]):
        raise ValueError("run_id must be non-empty and distinct from analysis_id.")
    if not isinstance(config["methods"], list) or not config["methods"]:
        raise ValueError("methods must be a non-empty list.")
    if config["mode"] == "simulation" and "simulation" not in config:
        raise ValueError("simulation mode requires a simulation section.")
    if config["mode"] == "simulation" and config.get("truth_source", "population") not in {
        "injected", "population", "observed",
    }:
        raise ValueError("truth_source must be injected, population, or observed.")
    if config["mode"] == "real_data" and "real_data" not in config:
        raise ValueError("real_data mode requires a real_data section.")
    if config["mode"] == "real_data" and not (
        config["real_data"].get("input_csv") or config["real_data"].get("prepared_manifest")
    ):
        raise ValueError("real_data requires input_csv or prepared_manifest.")
    if config.get("decision_rule_version", "v1") != "v1":
        raise ValueError("Only decision_rule_version='v1' is registered.")


def _add_run_columns(canonical: CanonicalDAInput, run_id: str, analysis_id: str) -> CanonicalDAInput:
    frames = []
    for frame in (
        canonical.abundance_long,
        canonical.sample_manifest,
        canonical.cell_type_manifest,
        canonical.contrast_specification,
    ):
        enriched = frame.copy()
        enriched["run_id"] = run_id
        enriched["analysis_id"] = analysis_id
        frames.append(enriched)
    return CanonicalDAInput(*frames).validate()


def _contrast_frame(config: dict[str, Any]) -> pd.DataFrame:
    contrast = config["contrast"]
    required = {"contrast_id", "factor", "group_1", "group_2"}
    if missing := required - set(contrast):
        raise ValueError(f"Contrast configuration is missing: {sorted(missing)}")
    reference = contrast.get("reference_cell_type")
    return pd.DataFrame([{
        "contrast_id": contrast["contrast_id"],
        "contrast_type": "pairwise",
        "contrast_definition": contrast.get(
            "contrast_definition", f"{contrast['group_1']} - {contrast['group_2']}"
        ),
        "factor": contrast["factor"],
        "group_1": contrast["group_1"],
        "group_2": contrast["group_2"],
        "reference_group": contrast.get("reference_group", contrast["group_2"]),
        "reference_cell_type": reference,
        "reference_strategy": "common_exclusion" if reference else "not_applicable",
        "reference_selection_reason": contrast.get("reference_selection_reason", pd.NA),
        "reference_is_fixed": bool(reference),
    }])


def _canonical_from_simulation(
    config: dict[str, Any],
    *,
    analysis_id: str,
    seed: int,
) -> tuple[CanonicalDAInput, pd.DataFrame]:
    simulation = config["simulation"]
    if simulation.get("generator", "dirichlet_multinomial") != "dirichlet_multinomial":
        raise ValueError("The phase-2 minimal pipeline supports the existing DM generator only.")
    parameters = dict(simulation.get("parameters", {}))
    parameters["random_state"] = seed
    reference = config["contrast"].get("reference_cell_type")
    if reference:
        protected = list(parameters.get("protected_cell_types", []))
        parameters["protected_cell_types"] = list(dict.fromkeys([*protected, reference]))
        parameters.setdefault("population_reference_cell_type", reference)
    abundance_source, legacy_truth = simulate_DM_data(**parameters)
    contrast = config["contrast"]
    factor = str(contrast["factor"])
    if factor not in abundance_source.columns:
        raise ValueError(f"Simulation output does not contain contrast factor {factor!r}.")
    group_counts = abundance_source.groupby(factor)["sample_id"].nunique()
    for group in (contrast["group_1"], contrast["group_2"]):
        if int(group_counts.get(group, 0)) < 2:
            raise ValueError(f"Simulation seed {seed} produced fewer than two samples for {group!r}.")

    abundance = abundance_source[["sample_id", "cell_type", "count", "total_count", "prop"]].copy()
    abundance = abundance.rename(columns={"prop": "proportion"})
    sample_columns = [
        column for column in (
            "sample_id", "donor_id", factor, "tissue", "assignment_strategy",
        )
        if column in abundance_source
    ]
    samples = abundance_source[sample_columns].drop_duplicates("sample_id").copy()
    samples["inclusion_status"] = "included"
    cell_types = pd.DataFrame({
        "cell_type": sorted(abundance["cell_type"].astype(str).unique()),
        "inclusion_status": "included",
    })
    canonical = CanonicalDAInput(abundance, samples, cell_types, _contrast_frame(config))
    canonical = _add_run_columns(canonical, str(config["run_id"]), analysis_id)

    if reference not in set(cell_types["cell_type"]):
        raise ValueError(f"Configured common reference cell type is absent: {reference!r}")
    target_truth = legacy_truth.loc[
        legacy_truth["contrast_factor"].eq(factor)
        & legacy_truth["contrast_group"].astype(str).eq(str(contrast["group_1"]))
        & legacy_truth["contrast_ref"].astype(str).eq(str(contrast["group_2"]))
    ].copy()
    if target_truth.empty:
        raise ValueError("The existing generator did not emit truth for the configured contrast.")
    source_columns = {
        "injected": (
            "Injected_Effect", "Is_Injected_Nonzero", "Injected_Effect_Scale", None,
            "model_native_net_composition_contrast",
        ),
        "population": (
            "Population_Effect", "Is_Population_Nonzero", "Population_Effect_Scale",
            "Population_Effect_Status", "Population_Effect_Estimand",
        ),
        "observed": (
            "Observed_Effect", "Is_Observed_Detectable", "Observed_Effect_Scale",
            "Observed_Effect_Status", "empirical_marginal_proportion_log2_fold_change",
        ),
    }
    truth_rows: list[dict[str, Any]] = []
    for _, legacy in target_truth.iterrows():
        for source, (effect_col, flag_col, scale_col, status_col, estimand) in source_columns.items():
            status = str(legacy.get(status_col, "available")) if status_col else "available"
            effect_estimand = legacy.get(estimand, pd.NA) if estimand in legacy.index else estimand
            row = {
                "run_id": str(config["run_id"]),
                "analysis_id": analysis_id,
                "scenario_id": simulation["scenario_id"],
                "replicate_id": analysis_id.rsplit("-", 1)[-1],
                "simulation_seed": seed,
                "contrast_id": contrast["contrast_id"],
                "cell_type": legacy["cell_type"],
                "effect_component": "composition",
                "truth_source": source,
                "is_true_effect": legacy.get(flag_col, pd.NA),
                "effect_value": legacy.get(effect_col, pd.NA),
                "effect_estimand": effect_estimand,
                "effect_scale": legacy.get(scale_col, pd.NA),
                "reference_cell_type": reference,
                "truth_status": status,
                "assignment_strategy": simulation.get("parameters", {}).get(
                    "assignment_strategy", "balanced"
                ),
                "is_closure_induced": (
                    legacy.get("Is_Population_Closure_Induced", pd.NA)
                    if source == "population" else False
                ),
                "null_tolerance": (
                    legacy.get("Population_Null_Tolerance", pd.NA)
                    if source == "population" else 0.0
                ),
                "Injected_Effect": legacy.get("Injected_Effect", pd.NA),
                "Population_Effect": legacy.get("Population_Effect", pd.NA),
                "Observed_Effect": legacy.get("Observed_Effect", pd.NA),
                "Is_Injected_Nonzero": legacy.get("Is_Injected_Nonzero", pd.NA),
                "Is_Population_Nonzero": legacy.get("Is_Population_Nonzero", pd.NA),
                "Is_Observed_Detectable": legacy.get("Is_Observed_Detectable", pd.NA),
            }
            truth_rows.append(row)
    return canonical, validate_truth_table(pd.DataFrame(truth_rows))


def _canonical_from_prepared_manifest(
    config: dict[str, Any],
    manifest_path: str | Path,
) -> tuple[CanonicalDAInput, dict[str, Any]]:
    path = Path(manifest_path).resolve()
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "phase4-step08-preparation-v1":
        raise ValueError("Unsupported prepared real-data manifest version.")
    run_root = path.parent.parent
    required_tables = {
        "abundance.csv", "sample_manifest.csv", "cell_type_manifest.csv",
        "contrast_specification.csv",
    }
    if missing := required_tables - set(manifest.get("tables", {})):
        raise ValueError(f"Prepared real-data manifest is missing tables: {sorted(missing)}")
    frames: dict[str, pd.DataFrame] = {}
    for name in required_tables:
        record = manifest["tables"][name]
        table_path = run_root / record["path"]
        content = table_path.read_bytes()
        if len(content) != int(record["bytes"]) or sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Prepared real-data table failed integrity verification: {name}")
        frames[name] = pd.read_csv(table_path)
    prepared = CanonicalDAInput(
        frames["abundance.csv"],
        frames["sample_manifest.csv"],
        frames["cell_type_manifest.csv"],
        frames["contrast_specification.csv"],
    ).validate()
    hash_algorithm = manifest.get(
        "canonical_input_hash_algorithm", "canonical-csv-sha256-v1"
    )
    if prepared.input_hash(hash_algorithm) != manifest["canonical_input_hash"]:
        raise ValueError("Prepared real-data canonical input hash does not match its manifest.")
    expected_contrast = _contrast_frame(config).iloc[0]
    actual_contrast = prepared.contrast_specification.iloc[0]
    for column in ("contrast_id", "factor", "group_1", "group_2", "reference_cell_type"):
        if str(actual_contrast.get(column)) != str(expected_contrast.get(column)):
            raise ValueError(f"Prepared real-data contrast disagrees with config at {column!r}.")
    canonical = _add_run_columns(
        prepared, str(config["run_id"]), str(config["analysis_id"])
    )
    qc_record = manifest["tables"].get("input_qc.csv")
    zero_filled = 0
    if qc_record:
        qc = pd.read_csv(run_root / qc_record["path"])
        zero_filled = int(qc.iloc[0].get("zero_filled_sample_cell_pairs", 0))
    return canonical, {
        "zero_filled_sample_cell_pairs": zero_filled,
        "prepared_manifest": str(path),
        "preparation_run_id": manifest["run_id"],
        "preparation_analysis_id": manifest["analysis_id"],
        "preparation_input_hash": manifest["canonical_input_hash"],
        "preparation_input_hash_algorithm": hash_algorithm,
    }


def _canonical_from_real_data(config: dict[str, Any]) -> tuple[CanonicalDAInput, dict[str, Any]]:
    real = config["real_data"]
    if real.get("prepared_manifest"):
        return _canonical_from_prepared_manifest(config, real["prepared_manifest"])
    source_path = Path(real["input_csv"])
    sample_column = real.get("sample_column", "orig.ident")
    cell_column = real.get("cell_type_column", "Celltype")
    group_column = real.get("group_column", config["contrast"]["factor"])
    samples_per_group = int(real.get("samples_per_group", 4))
    extra_columns = list(real.get("sample_metadata_columns", []))
    columns = list(dict.fromkeys([sample_column, cell_column, group_column, *extra_columns]))
    source = pd.read_csv(source_path, usecols=columns)
    source[sample_column] = source[sample_column].astype(str)
    source[cell_column] = source[cell_column].astype(str)
    group_1 = config["contrast"]["group_1"]
    group_2 = config["contrast"]["group_2"]
    cohort = source.loc[source[group_column].isin([group_1, group_2])].copy()
    metadata_columns = [sample_column, group_column, *extra_columns]
    metadata = cohort[metadata_columns].drop_duplicates()
    if metadata.groupby(sample_column).size().gt(1).any():
        raise ValueError("Configured real-data sample metadata is not sample-constant.")
    selected: list[str] = []
    for group in (group_1, group_2):
        candidates = sorted(metadata.loc[metadata[group_column].eq(group), sample_column].tolist())
        if len(candidates) < samples_per_group:
            raise ValueError(f"Group {group!r} has fewer than {samples_per_group} samples.")
        selected.extend(candidates[:samples_per_group])
    cohort = cohort.loc[cohort[sample_column].isin(selected)]
    cell_types = sorted(cohort[cell_column].unique())
    observed = cohort.groupby([sample_column, cell_column], observed=True).size().rename("count")
    index = pd.MultiIndex.from_product([selected, cell_types], names=[sample_column, cell_column])
    zero_filled = int(len(index.difference(observed.index)))
    abundance = observed.reindex(index, fill_value=0).reset_index().rename(columns={
        sample_column: "sample_id", cell_column: "cell_type",
    })
    abundance["total_count"] = abundance.groupby("sample_id")["count"].transform("sum")
    abundance["proportion"] = abundance["count"] / abundance["total_count"]
    samples = (
        metadata.loc[metadata[sample_column].isin(selected)]
        .rename(columns={sample_column: "sample_id", group_column: config["contrast"]["factor"]})
        .set_index("sample_id").loc[selected].reset_index()
    )
    samples["inclusion_status"] = "included"
    cell_manifest = pd.DataFrame({"cell_type": cell_types, "inclusion_status": "included"})
    reference = config["contrast"].get("reference_cell_type")
    if reference not in set(cell_types):
        raise ValueError(f"Configured common reference cell type is absent: {reference!r}")
    canonical = CanonicalDAInput(abundance, samples, cell_manifest, _contrast_frame(config))
    canonical = _add_run_columns(canonical, str(config["run_id"]), str(config["analysis_id"]))
    return canonical, {"zero_filled_sample_cell_pairs": zero_filled}


def _decision_registry(methods: list[str]):
    registry = load_default_decision_rules()
    if "mock_frequentist" in methods:
        registry.register(DecisionRule(
            "mock-frequentist-primary-v1", "pvalue_adjusted", "<", 0.05,
            "Mock adjusted p-value rule", method="mock_frequentist",
        ))
    if "mock_bayesian" in methods:
        registry.register(DecisionRule(
            "mock-bayesian-primary-v1", "posterior_inclusion_probability", ">=", 0.95,
            "Mock PIP rule", method="mock_bayesian",
        ))
    return registry


def _build_adapters(config: dict[str, Any], run_root: Path):
    methods = list(config["methods"])
    runtime = config.get("runtime", {})
    r_methods = {"propeller", "dcats", "sccomp"} & set(methods)
    bridge = None
    if r_methods:
        bridge = RScriptBridge(
            Path(runtime.get("rscript", "C:/Program Files/R/R-4.6.1/bin/Rscript.exe")),
            library_path=Path(runtime.get("r_library", "environment/R/library")).resolve(),
            cmdstan_path=(
                Path(runtime["cmdstan_path"]).resolve() if runtime.get("cmdstan_path") else None
            ),
            staging_root=run_root / "staging：临时目录",
            timeout_seconds=int(runtime.get("timeout_seconds", 3600)),
        )
    adapters = []
    for method in methods:
        if method == "propeller":
            adapters.append(PropellerAdapter(method_version="1.12.0", bridge=bridge))
        elif method == "dcats":
            adapters.append(DCATSAdapter(method_version="1.10.0", bridge=bridge))
        elif method == "sccomp":
            cache = runtime.get("sccomp_model_cache", str(run_root / "sccomp_model_cache"))
            adapters.append(SccompAdapter(
                method_version="2.4.0",
                bridge=bridge,
                cores=int(runtime.get("sccomp_cores", 2)),
                model_cache_dir=str(Path(cache).resolve()),
                max_sampling_iterations=int(runtime.get("sccomp_max_sampling_iterations", 1000)),
                include_variability=bool(runtime.get("sccomp_include_variability", True)),
            ))
        elif method == "sccoda":
            adapters.append(ScCODAAdapter(
                method_version="pertpy-1.1.1",
                reference_cell_type=config["contrast"].get("reference_cell_type"),
                num_samples=int(runtime.get("sccoda_num_samples", 1000)),
                num_warmup=int(runtime.get("sccoda_num_warmup", 500)),
                num_chains=int(runtime.get("sccoda_num_chains", 2)),
                rng_key=int(runtime.get("sccoda_rng_key", 7)),
            ))
        elif method == "tri_anchor":
            tri_config = dict(config.get("tri_anchor", {}))
            rule_path = tri_config.pop("rule_path", None)
            rule = load_tri_anchor_rule(rule_path, overrides=tri_config)
            missing_anchors = set(rule.anchor_methods) - set(methods)
            if missing_anchors:
                raise ValueError(
                    "Tri_anchor configured anchor methods must also be pipeline methods: "
                    f"{sorted(missing_anchors)}"
                )
            adapters.append(TriAnchorAdapter(rule=rule))
        elif method == "mock_frequentist":
            adapters.append(MockFrequentistAdapter())
        elif method == "mock_bayesian":
            adapters.append(MockBayesianAdapter())
        elif method == "mock_failure":
            adapters.append(MockFailureAdapter())
        elif method == "naive_welch_proportion":
            adapters.append(NaiveWelchProportionAdapter(method_version="scipy"))
        else:
            raise ValueError(f"Unsupported pipeline method: {method!r}")
    return adapters


def _save_input(canonical: CanonicalDAInput, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    tables = {
        "abundance.csv": canonical.abundance_long,
        "sample_manifest.csv": canonical.sample_manifest,
        "cell_type_manifest.csv": canonical.cell_type_manifest,
        "contrast_specification.csv": canonical.contrast_specification,
    }
    for name, frame in tables.items():
        frame.to_csv(target / name, index=False)
    (target / "input_manifest.json").write_text(json.dumps({
        "run_id": str(canonical.abundance_long["run_id"].iloc[0]),
        "analysis_id": str(canonical.abundance_long["analysis_id"].iloc[0]),
        "input_hash": canonical.input_hash(),
        "tables": list(tables),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _real_data_summaries(public: pd.DataFrame, reference_cell_type: str) -> dict[str, pd.DataFrame]:
    composition = public.loc[public["effect_component"].eq("composition")].copy()
    reference = composition["cell_type"].astype(str).eq(str(reference_cell_type))
    candidate = composition.loc[~reference].copy()
    candidate["primary_decision"] = parse_boolean_series(candidate["primary_decision"])
    method_rows = []
    for method, group in candidate.groupby("method", sort=False):
        method_rows.append({
            "run_id": group["run_id"].iloc[0],
            "analysis_id": group["analysis_id"].iloc[0],
            "method": method,
            "tested_universe": len(group),
            "number_tested": int(group["contrast_status"].eq("success").sum()),
            "discoveries": int(group["primary_decision"].fillna(False).sum()),
            "number_invalid": int(group["contrast_status"].eq("invalid").sum()),
            "number_unavailable": int((~parse_boolean_series(group["is_available"]).fillna(False)).sum()),
            "number_reference_excluded": int((reference & composition["method"].eq(method)).sum()),
        })
    method_summary = pd.DataFrame(method_rows)
    direction_summary = (
        candidate.groupby(["run_id", "analysis_id", "method", "effect_direction"], dropna=False)
        .size().reset_index(name="number_results")
    )
    pivot = candidate.pivot_table(
        index=["run_id", "analysis_id", "contrast_id", "cell_type"],
        columns="method",
        values="primary_decision",
        aggfunc="first",
        dropna=False,
    )
    complete = pivot.dropna().copy()
    agreement = complete.nunique(axis=1).eq(1)
    method_agreement = complete.reset_index()
    method_agreement["all_methods_agree"] = agreement.to_numpy()
    method_agreement["positive_method_count"] = complete.astype(bool).sum(axis=1).to_numpy()
    contrast_summary = (
        candidate.groupby(["run_id", "analysis_id", "method", "contrast_id"], dropna=False)
        .agg(
            tested_universe=("cell_type", "size"),
            discoveries=("primary_decision", lambda values: int(parse_boolean_series(values).fillna(False).sum())),
            mean_estimate=("estimate", "mean"),
        ).reset_index()
    )
    return {
        "method_summary": method_summary,
        "effect_direction_summary": direction_summary,
        "method_agreement": method_agreement,
        "contrast_summary": contrast_summary,
        "plot_ready_real": method_summary.copy(),
    }


def _write_tables(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False)


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(file for file in root.rglob("*") if file.is_file() and file.name != "run_manifest.json"):
        content = path.read_bytes()
        records.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(content),
            "sha256": sha256(content).hexdigest(),
        })
    return records


def _configure_plot_cache(config: dict[str, Any], output_root: Path) -> None:
    """Keep mutable Matplotlib caches outside immutable run directories."""
    runtime = config.get("runtime", {})
    configured = runtime.get("matplotlib_cache")
    cache = Path(configured).resolve() if configured else output_root / ".runtime_cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def run_abundance_pipeline(config: str | Path | Mapping[str, Any]) -> PipelineRunResult:
    """Run the sole formal simulation or real-data differential-abundance pipeline."""
    document = _load_config(config)
    _validate_config(document)
    run_id = str(document["run_id"])
    mode = str(document["mode"])
    output_root = Path(document["output_root"]).resolve()
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    (run_root / "config_snapshot.yaml").write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "analysis_id": str(document["analysis_id"]),
        "mode": mode,
        "methods": list(document["methods"]),
        "truth_source": str(document.get("truth_source", "population")) if mode == "simulation" else None,
        "decision_rule_version": document.get("decision_rule_version", "v1"),
        "status": "running",
        "started_at": started_at,
    }
    try:
        adapters = _build_adapters(document, run_root)
        registry = _decision_registry(list(document["methods"]))
        public_frames = []
        evidence_frames = []
        diagnostic_frames = []
        truth_frames = []
        input_notes: dict[str, Any] = {}

        if mode == "simulation":
            simulation = document["simulation"]
            replicates = int(simulation.get("replicates", 1))
            base_seed = int(simulation["simulation_seed"])
            for index in range(1, replicates + 1):
                analysis_id = f"{document['analysis_id']}-r{index:03d}"
                seed = base_seed + index - 1
                canonical, truth = _canonical_from_simulation(
                    document, analysis_id=analysis_id, seed=seed
                )
                _save_input(canonical, run_root / "inputs" / analysis_id)
                result = DifferentialAbundanceRunner(
                    run_root / "analyses" / analysis_id, registry
                ).run(canonical, adapters, run_id=run_id, analysis_id=analysis_id)
                public_frames.append(result.public_view)
                evidence_frames.append(result.evidence_layer)
                diagnostic_frames.append(result.diagnostics)
                truth_frames.append(truth)
        else:
            canonical, real_input_notes = _canonical_from_real_data(document)
            analysis_id = str(document["analysis_id"])
            input_notes.update(real_input_notes)
            _save_input(canonical, run_root / "inputs" / analysis_id)
            result = DifferentialAbundanceRunner(
                run_root / "analyses" / analysis_id, registry
            ).run(canonical, adapters, run_id=run_id, analysis_id=analysis_id)
            public_frames.append(result.public_view)
            evidence_frames.append(result.evidence_layer)
            diagnostic_frames.append(result.diagnostics)

        public = pd.concat(public_frames, ignore_index=True)
        evidence = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame()
        diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
        canonical_dir = run_root / "canonical"
        canonical_dir.mkdir()
        public.to_csv(canonical_dir / "contrast_public.csv", index=False)
        evidence.to_csv(canonical_dir / "evidence.csv", index=False)
        diagnostics.to_csv(canonical_dir / "diagnostics.csv", index=False)

        evaluation = None
        truth_table = None
        if mode == "simulation":
            truth_table = validate_truth_table(pd.concat(truth_frames, ignore_index=True))
            truth_dir = run_root / "truth"
            truth_dir.mkdir()
            truth_table.to_csv(truth_dir / "truth_table.csv", index=False)
            selected_source = str(document.get("truth_source", "population"))
            selected_truth = truth_table.loc[
                truth_table["truth_source"].eq(selected_source)
                & truth_table["effect_component"].eq("composition")
            ]
            selected_flags = parse_boolean_series(selected_truth["is_true_effect"])
            closure_flags = (
                parse_boolean_series(selected_truth["is_closure_induced"])
                if "is_closure_induced" in selected_truth.columns else
                pd.Series(False, index=selected_truth.index, dtype="boolean")
            )
            truth_definition = {
                "truth_source": selected_source,
                "effect_estimands": sorted(selected_truth["effect_estimand"].dropna().astype(str).unique()),
                "effect_scales": sorted(selected_truth["effect_scale"].dropna().astype(str).unique()),
                "reference_cell_types": sorted(
                    selected_truth["reference_cell_type"].dropna().astype(str).unique()
                ),
                "null_tolerances": sorted(
                    pd.to_numeric(selected_truth.get("null_tolerance"), errors="coerce")
                    .dropna().astype(float).unique().tolist()
                    if "null_tolerance" in selected_truth.columns else []
                ),
                "assignment_strategies": sorted(
                    selected_truth.get("assignment_strategy", pd.Series(dtype="object"))
                    .dropna().astype(str).unique()
                ),
                "number_rows": len(selected_truth),
                "number_true_effects": int(selected_flags.eq(True).sum()),
                "number_null_effects": int(selected_flags.eq(False).sum()),
                "number_closure_induced": int(closure_flags.eq(True).sum()),
                "binary_benchmark_informative": bool(
                    selected_flags.eq(True).any() and selected_flags.eq(False).any()
                ),
            }
            (truth_dir / "truth_manifest.json").write_text(
                json.dumps(truth_definition, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest["truth_definition"] = truth_definition
            evaluation = evaluate_contrasts(
                public,
                truth_table,
                EvaluationSpec(
                    truth_source=selected_source,
                    methods=tuple(str(method) for method in document["methods"]),
                ),
            )
            summary_tables = {
                "aligned_evaluation": evaluation.aligned,
                "method_completion": evaluation.method_completion,
                "replicate_metrics": evaluation.replicate_metrics,
                "aggregate_metrics": evaluation.aggregate_metrics,
                "plot_ready_evaluation": evaluation.aggregate_metrics.copy(),
            }
            _write_tables(summary_tables, run_root / "evaluation")
            figures = run_root / "figures"
            figures.mkdir()
            _configure_plot_cache(document, output_root)
            from src.stats.plot.plotting_helpers import plot_evaluation_summary
            plot_evaluation_summary(
                summary_tables["plot_ready_evaluation"], figures / "evaluation_summary.png"
            )
        else:
            summary_tables = _real_data_summaries(
                public, str(document["contrast"]["reference_cell_type"])
            )
            _write_tables(summary_tables, run_root / "summaries")
            figures = run_root / "figures"
            figures.mkdir()
            _configure_plot_cache(document, output_root)
            from src.stats.plot.plotting_helpers import plot_real_data_summary
            plot_real_data_summary(summary_tables["plot_ready_real"], figures / "real_data_summary.png")

        manifest.update({
            "status": "success",
            "finished_at": _utc_now(),
            "analysis_ids": sorted(public["analysis_id"].astype(str).unique()),
            "input_notes": input_notes,
            "files": _file_manifest(run_root),
        })
        (run_root / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return PipelineRunResult(
            run_id, mode, run_root, public, evidence, diagnostics,
            truth_table, evaluation, summary_tables,
        )
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "finished_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "files": _file_manifest(run_root),
        })
        (run_root / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical abundance pipeline.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = run_abundance_pipeline(args.config)
    print(result.output_root)


if __name__ == "__main__":
    main()
