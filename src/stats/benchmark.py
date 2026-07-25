from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.stats.evaluation import EvaluationSpec, evaluate_contrasts
from src.stats.pipeline import _build_adapters, _canonical_from_simulation, _decision_registry
from src.stats.runners import DifferentialAbundanceRunner
from src.stats.schemas import CanonicalDAInput, validate_truth_table
from src.stats.validation import parse_boolean_series


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "config" / "method_benchmark_registry.yaml"
DEFAULT_SCENARIOS = REPOSITORY_ROOT / "config" / "phase5_scenario_matrix.csv"
DEFAULT_EVALUATION = REPOSITORY_ROOT / "config" / "phase5_evaluation_spec.yaml"
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "phase5_benchmark.yaml"
ALLOWED_TASK_STATUSES = {
    "pending", "running", "success", "runtime_failed", "diagnostics_invalid",
    "conversion_failed", "skipped_with_reason",
}
REGISTRY_FIELDS = {
    "method_id", "implementation", "benchmark_role", "scientific_status",
    "expected_estimand", "decision_rule_id", "required_environment", "enabled",
    "failure_policy", "notes",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_records(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_task_manifest(frame: pd.DataFrame, path: Path) -> None:
    if not set(frame["status"].astype(str)).issubset(ALLOWED_TASK_STATUSES):
        raise ValueError("Task manifest contains an unsupported status.")
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            # Windows virus scanners/indexers can briefly hold the destination.
            # This retries only the manifest rename, never a statistical method.
            time.sleep(0.2 * (attempt + 1))


def load_method_registry(path: str | Path = DEFAULT_REGISTRY) -> pd.DataFrame:
    source = Path(path)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    records = document.get("methods", []) if isinstance(document, dict) else []
    if not records:
        raise ValueError("Method benchmark registry is empty.")
    frame = pd.DataFrame(records)
    if missing := REGISTRY_FIELDS - set(frame.columns):
        raise ValueError(f"Method registry is missing fields: {sorted(missing)}")
    if frame["method_id"].duplicated().any():
        raise ValueError("Method registry method_id values must be unique.")
    frame["enabled"] = parse_boolean_series(frame["enabled"])
    if frame["enabled"].isna().any():
        raise ValueError("Method registry enabled values must be explicit booleans.")
    tri = frame.loc[frame["method_id"].eq("tri_anchor")]
    if tri.empty or bool(tri.iloc[0]["enabled"]):
        raise ValueError("Tri_anchor must be explicitly registered and disabled in Phase 5.")
    enabled_roles = set(frame.loc[frame["enabled"].astype(bool), "benchmark_role"])
    if not enabled_roles.issubset({"formal_candidate", "sanity_check"}):
        raise ValueError("Enabled methods must be formal_candidate or sanity_check.")
    sanity = frame.loc[frame["benchmark_role"].eq("sanity_check")]
    if sanity.empty or not sanity["scientific_status"].eq("intentionally_misspecified").all():
        raise ValueError("Sanity-check methods must be intentionally_misspecified.")
    return frame


def load_scenario_matrix(path: str | Path = DEFAULT_SCENARIOS) -> pd.DataFrame:
    # `null` is a registered scenario label, not a missing-value token.
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "scenario_id", "sample_size", "effect_strength", "difficulty", "n_donors",
        "n_samples_per_donor", "n_celltypes", "baseline_alpha_scale",
        "disease_effect_size", "tissue_effect_size", "interaction_effect_size",
        "inflamed_cell_frac", "sampling_bias_strength", "total_count_mean",
        "total_count_sd", "min_count", "donor_noise_sd", "reference_cell_type",
        "pilot_enabled", "production_enabled",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Scenario matrix is missing fields: {sorted(missing)}")
    if frame["scenario_id"].duplicated().any():
        raise ValueError("Scenario IDs must be unique.")
    for column in ("pilot_enabled", "production_enabled"):
        frame[column] = parse_boolean_series(frame[column])
        if frame[column].isna().any():
            raise ValueError(f"{column} must contain explicit booleans.")
    if not {"small", "medium", "large"}.issubset(set(frame["sample_size"])):
        raise ValueError("Scenario matrix must include small, medium, and large sample sizes.")
    if not {"null", "weak", "moderate", "strong"}.issubset(set(frame["effect_strength"])):
        raise ValueError("Scenario matrix must include null, weak, moderate, and strong effects.")
    if not frame.loc[frame["effect_strength"].eq("null"), "disease_effect_size"].eq(0).all():
        raise ValueError("Null scenarios must have zero disease effect.")
    for _, row in frame.iterrows():
        reference = str(row["reference_cell_type"])
        if not reference:
            raise ValueError("Every scenario requires a non-empty protected reference cell type.")
        if "cell_type_names_json" in frame and str(row.get("cell_type_names_json", "")):
            names = json.loads(str(row["cell_type_names_json"]))
            if reference not in set(map(str, names)):
                raise ValueError("Scenario reference is absent from cell_type_names_json.")
        elif reference != "CT" + str(row["n_celltypes"]):
            raise ValueError("Legacy generic scenarios must protect the final CT cell type.")
    return frame


def load_benchmark_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("generator") != "dirichlet_multinomial":
        raise ValueError("Phase 5 formal benchmark must use the DM generator.")
    if int(document.get("execution", {}).get("automatic_retries", -1)) != 0:
        raise ValueError("Phase 5 does not silently retry failed tasks.")
    return document


def _scenario_parameters(row: pd.Series) -> dict[str, Any]:
    parameters = {
        "n_donors": int(row["n_donors"]),
        "n_samples_per_donor": int(row["n_samples_per_donor"]),
        "n_celltypes": int(row["n_celltypes"]),
        "baseline_alpha_scale": float(row["baseline_alpha_scale"]),
        "disease_effect_size": float(row["disease_effect_size"]),
        "tissue_effect_size": float(row["tissue_effect_size"]),
        "interaction_effect_size": float(row["interaction_effect_size"]),
        "inflamed_cell_frac": float(row["inflamed_cell_frac"]),
        "sampling_bias_strength": float(row["sampling_bias_strength"]),
        "total_count_mean": float(row["total_count_mean"]),
        "total_count_sd": float(row["total_count_sd"]),
        "min_count": int(row["min_count"]),
        "donor_noise_sd": float(row["donor_noise_sd"]),
        "assignment_strategy": "balanced",
        "protected_cell_types": [str(row["reference_cell_type"])],
        "population_reference_cell_type": str(row["reference_cell_type"]),
        "population_null_tolerance": 1e-12,
        "disease_levels": ["control", "case"],
        "tissue_levels": ["nif", "if"],
    }
    if str(row.get("cell_type_names_json", "")):
        parameters["cell_type_names"] = json.loads(str(row["cell_type_names_json"]))
    if str(row.get("baseline_composition_json", "")):
        parameters["baseline_composition"] = json.loads(
            str(row["baseline_composition_json"])
        )
    return parameters


def _run_config_path(root: Path) -> Path:
    current = root / "benchmark_config" / "benchmark_config.yaml"
    return current if current.is_file() else root / "benchmark_config" / "phase5_benchmark.yaml"


def _phase_label_from_config(config: dict[str, Any]) -> str:
    schema_version = str(config.get("schema_version", "phase5-benchmark-v1"))
    label = schema_version.split("-", 1)[0]
    return label if label.startswith("phase") else "phase5"


def _phase_label_for_run(root: Path) -> str:
    config = yaml.safe_load(_run_config_path(root).read_text(encoding="utf-8"))
    return _phase_label_from_config(config)


def _replicate_paths(root: Path, scenario_id: str, replicate_id: str) -> dict[str, Path]:
    base = root / "scenarios" / scenario_id / replicate_id
    return {
        "root": base,
        "simulation": base / "simulation",
        "truth": base / "truth",
        "methods": base / "methods",
        "manifests": base / "manifests",
    }


def _save_frozen_replicate(
    benchmark_root: Path,
    benchmark_id: str,
    scenario: pd.Series,
    replicate_number: int,
    benchmark_config: dict[str, Any],
    phase: str,
) -> tuple[str, str, str, int]:
    scenario_id = str(scenario["scenario_id"])
    replicate_id = f"r{replicate_number:03d}"
    run_id = f"{benchmark_id}:{scenario_id}:{replicate_id}"
    analysis_id = f"{scenario_id}-{replicate_id}"
    seed_strategy = str(benchmark_config.get("seed_strategy", "offset"))
    if seed_strategy == "sha256_benchmark_scenario_replicate":
        excluded: set[int] = set()
        excluded_path = benchmark_config.get("excluded_seed_manifest")
        if excluded_path:
            excluded_frame = pd.read_csv(excluded_path)
            if "simulation_seed" not in excluded_frame:
                raise ValueError("excluded_seed_manifest requires simulation_seed.")
            excluded = set(pd.to_numeric(
                excluded_frame["simulation_seed"], errors="raise"
            ).astype(int))
        attempt = 0
        while True:
            suffix = "" if attempt == 0 else f"|retry={attempt}"
            token = f"{benchmark_id}|{scenario_id}|{replicate_id}{suffix}"
            seed = int(sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            if seed not in excluded:
                break
            attempt += 1
    elif seed_strategy == "offset":
        phase_offset = 0 if phase == "pilot" else 100_000
        seed = (
            int(benchmark_config["base_seed"])
            + phase_offset
            + int(scenario.name) * 1000
            + replicate_number
        )
    else:
        raise ValueError(f"Unknown benchmark seed_strategy: {seed_strategy!r}")
    paths = _replicate_paths(benchmark_root, scenario_id, replicate_id)
    paths["root"].mkdir(parents=True, exist_ok=False)
    for key in ("simulation", "truth", "methods", "manifests"):
        paths[key].mkdir()

    contrast = dict(benchmark_config["contrast"])
    contrast["reference_cell_type"] = str(scenario["reference_cell_type"])
    document = {
        "mode": "simulation",
        "run_id": run_id,
        "analysis_id": analysis_id,
        "methods": ["naive_welch_proportion"],
        "output_root": str(benchmark_root),
        "truth_source": "population",
        "contrast": contrast,
        "simulation": {
            "generator": "dirichlet_multinomial",
            "scenario_id": scenario_id,
            "simulation_seed": seed,
            "parameters": _scenario_parameters(scenario),
        },
    }
    config_path = paths["simulation"] / "simulation_config.yaml"
    config_path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    canonical, truth = _canonical_from_simulation(document, analysis_id=analysis_id, seed=seed)
    validate_truth_table(truth)
    canonical.abundance_long.to_csv(
        paths["simulation"] / "canonical_abundance_input.csv", index=False
    )
    canonical.sample_manifest.to_csv(paths["simulation"] / "sample_metadata.csv", index=False)
    canonical.cell_type_manifest.to_csv(
        paths["simulation"] / "cell_type_manifest.csv", index=False
    )
    canonical.contrast_specification.to_csv(
        paths["simulation"] / "contrast_specification.csv", index=False
    )
    # The immutable task hash is computed from the exact CSV handoff consumed by
    # methods, not from the pre-serialization in-memory frames.
    frozen_canonical = CanonicalDAInput(
        pd.read_csv(paths["simulation"] / "canonical_abundance_input.csv"),
        pd.read_csv(paths["simulation"] / "sample_metadata.csv"),
        pd.read_csv(paths["simulation"] / "cell_type_manifest.csv"),
        pd.read_csv(paths["simulation"] / "contrast_specification.csv"),
    ).validate()
    frozen_input_hash = frozen_canonical.input_hash()
    truth.to_csv(paths["truth"] / "truth_table.csv", index=False)
    truth.loc[truth["truth_source"].eq("injected")].to_csv(
        paths["truth"] / "injection_manifest.csv", index=False
    )
    population = truth.loc[truth["truth_source"].eq("population")].copy()
    population.to_csv(paths["truth"] / "population_truth_manifest.csv", index=False)

    abundance = canonical.abundance_long
    flags = parse_boolean_series(population["is_true_effect"])
    diagnostics = {
        "generator": "dirichlet_multinomial",
        "scenario_id": scenario_id,
        "replicate_id": replicate_id,
        "simulation_seed": seed,
        "input_hash": frozen_input_hash,
        "number_samples": int(canonical.sample_manifest["sample_id"].nunique()),
        "number_cell_types": int(canonical.cell_type_manifest["cell_type"].nunique()),
        "number_zero_counts": int(pd.to_numeric(abundance["count"]).eq(0).sum()),
        "maximum_proportion_sum_error": float(
            (abundance.groupby("sample_id")["proportion"].sum() - 1).abs().max()
        ),
        "number_population_true": int(flags.eq(True).sum()),
        "number_population_null": int(flags.eq(False).sum()),
        "reference_injected": bool(
            truth.loc[
                truth["truth_source"].eq("injected")
                & truth["cell_type"].astype(str).eq(str(scenario["reference_cell_type"])),
                "is_true_effect",
            ].fillna(False).any()
        ),
        "assignment_strategy": "balanced",
    }
    _write_json(paths["simulation"] / "generator_diagnostics.json", diagnostics)
    manifest = {
        "schema_version": (
            f"{_phase_label_from_config(benchmark_config)}-simulation-replicate-v1"
        ),
        "benchmark_id": benchmark_id,
        "run_id": run_id,
        "analysis_id": analysis_id,
        "scenario_id": scenario_id,
        "replicate_id": replicate_id,
        "simulation_seed": seed,
        "status": "frozen_before_method_execution",
        "input_hash": frozen_input_hash,
        "created_at": _utc_now(),
        "files": _file_records(paths["root"], excluded={"manifests/run_manifest.json"}),
    }
    _write_json(paths["manifests"] / "run_manifest.json", manifest)
    return run_id, analysis_id, frozen_input_hash, seed


def initialize_benchmark(
    benchmark_id: str,
    *,
    phase: str,
    replicates: int,
    output_base: str | Path = REPOSITORY_ROOT / "benchmark_runs",
    registry_path: str | Path = DEFAULT_REGISTRY,
    scenario_path: str | Path = DEFAULT_SCENARIOS,
    evaluation_path: str | Path = DEFAULT_EVALUATION,
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    if phase not in {"pilot", "production"}:
        raise ValueError("phase must be pilot or production.")
    if replicates < 1:
        raise ValueError("replicates must be positive.")
    registry = load_method_registry(registry_path)
    scenarios = load_scenario_matrix(scenario_path)
    config = load_benchmark_config(config_path)
    enabled_column = "pilot_enabled" if phase == "pilot" else "production_enabled"
    scenarios = scenarios.loc[scenarios[enabled_column].astype(bool)].reset_index(drop=True)
    root = Path(output_base).resolve() / benchmark_id
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "benchmark_config", "method_registry", "scenarios", "evaluation", "summaries",
        "figures", "diagnostics", "logs", "final_report",
    ):
        (root / name).mkdir()
    snapshots = {
        Path(config_path): root / "benchmark_config" / "benchmark_config.yaml",
        Path(scenario_path): root / "benchmark_config" / "scenario_matrix.csv",
        Path(evaluation_path): root / "benchmark_config" / "evaluation_spec.yaml",
        Path(registry_path): root / "method_registry" / "method_benchmark_registry.yaml",
    }
    for source, target in snapshots.items():
        shutil.copyfile(source, target)

    tasks: list[dict[str, Any]] = []
    for scenario_index, scenario in scenarios.iterrows():
        scenario = scenario.copy()
        scenario.name = scenario_index
        for replicate_number in range(1, replicates + 1):
            run_id, analysis_id, input_hash, seed = _save_frozen_replicate(
                root, benchmark_id, scenario, replicate_number, config, phase
            )
            for _, method in registry.iterrows():
                enabled = bool(method["enabled"])
                tasks.append({
                    "benchmark_id": benchmark_id,
                    "phase": phase,
                    "scenario_id": scenario["scenario_id"],
                    "replicate_id": f"r{replicate_number:03d}",
                    "run_id": run_id,
                    "analysis_id": analysis_id,
                    "simulation_seed": seed,
                    "method": method["method_id"],
                    "benchmark_role": method["benchmark_role"],
                    "scientific_status": method["scientific_status"],
                    "input_hash": input_hash,
                    "status": "pending" if enabled else "skipped_with_reason",
                    "failure_reason": pd.NA if enabled else method["notes"],
                    "attempt_count": 0,
                    "started_at": pd.NA,
                    "finished_at": pd.NA,
                    "runtime_seconds": pd.NA,
                    "result_subdir": ".",
                })
    task_frame = pd.DataFrame(tasks)
    task_path = root / "benchmark_task_manifest.csv"
    _write_task_manifest(task_frame, task_path)
    manifest = {
        "schema_version": f"{_phase_label_from_config(config)}-benchmark-run-v1",
        "benchmark_id": benchmark_id,
        "phase": phase,
        "replicates_per_scenario": replicates,
        "scenario_ids": scenarios["scenario_id"].astype(str).tolist(),
        "enabled_methods": registry.loc[registry["enabled"].astype(bool), "method_id"].tolist(),
        "status": "initialized_simulations_frozen",
        "created_at": _utc_now(),
        "automatic_retries": 0,
        "files": _file_records(
            root,
            excluded={"benchmark_manifest.json", "benchmark_task_manifest.csv"},
        ),
    }
    _write_json(root / "benchmark_manifest.json", manifest)
    return root


def derive_expanded_benchmark(
    source_benchmark_root: str | Path,
    benchmark_id: str,
    *,
    output_base: str | Path = REPOSITORY_ROOT / "benchmark_runs",
    registry_path: str | Path = REPOSITORY_ROOT / "config" / "phase6_method_benchmark_registry.yaml",
    config_path: str | Path = REPOSITORY_ROOT / "config" / "phase6_benchmark.yaml",
) -> Path:
    """Create a new immutable benchmark lineage from frozen inputs and completed outputs.

    The source tree is never modified. Frozen canonical inputs are copied byte-for-byte,
    existing method outputs are re-keyed to the new run lineage, and methods newly enabled
    by the supplied registry become pending tasks. Evaluation/report artifacts are not
    copied, preventing stale summaries from entering the expanded review package.
    """
    source = Path(source_benchmark_root).resolve()
    target = Path(output_base).resolve() / benchmark_id
    if target.exists():
        raise FileExistsError(f"Derived benchmark already exists: {target}")
    source_manifest = json.loads((source / "benchmark_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "stop_point_d_awaiting_human_review":
        raise RuntimeError("The source benchmark must be sealed at Stop Point D.")
    registry = load_method_registry(registry_path)
    enabled_methods = set(
        registry.loc[registry["enabled"].astype(bool), "method_id"].astype(str)
    )
    registry_lookup = registry.set_index("method_id")
    excluded_root_entries = {
        "evaluation", "summaries", "figures", "diagnostics", "logs", "final_report",
        "method_completion_matrix.csv", "missing_success_outputs.csv",
        "evaluation_readiness_report.md", "runtime_summary.csv", "method_failure_summary.csv",
    }

    def ignore_stale(path: str, names: list[str]) -> set[str]:
        return excluded_root_entries & set(names) if Path(path).resolve() == source else set()

    shutil.copytree(source, target, ignore=ignore_stale)
    for name in ("evaluation", "summaries", "figures", "diagnostics", "logs", "final_report"):
        (target / name).mkdir()
    shutil.copyfile(config_path, target / "benchmark_config" / "benchmark_config.yaml")
    shutil.copyfile(registry_path, target / "method_registry" / "method_benchmark_registry.yaml")

    tasks = pd.read_csv(target / "benchmark_task_manifest.csv")
    tasks["benchmark_id"] = benchmark_id
    for (scenario_id, replicate_id), indices in tasks.groupby(
        ["scenario_id", "replicate_id"], sort=False
    ).groups.items():
        new_run_id = f"{benchmark_id}:{scenario_id}:{replicate_id}"
        tasks.loc[list(indices), "run_id"] = new_run_id
        replicate = _replicate_paths(target, str(scenario_id), str(replicate_id))
        run_manifest_path = replicate["manifests"] / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        old_run_id = str(run_manifest["run_id"])
        run_manifest.update({
            "run_id": new_run_id,
            "derived_from_run_id": old_run_id,
            "derived_from_benchmark_id": source.name,
            "status": "frozen_input_reused_for_expanded_methods",
        })
        for truth_csv in replicate["truth"].glob("*.csv"):
            frame = pd.read_csv(truth_csv)
            if "run_id" in frame:
                frame["run_id"] = new_run_id
                frame.to_csv(truth_csv, index=False)
        for index in indices:
            method = str(tasks.at[index, "method"])
            if method not in registry_lookup.index:
                continue
            registered = registry_lookup.loc[method]
            tasks.at[index, "benchmark_role"] = registered["benchmark_role"]
            tasks.at[index, "scientific_status"] = registered["scientific_status"]
            if method in enabled_methods and str(tasks.at[index, "status"]) == "skipped_with_reason":
                tasks.at[index, "status"] = "pending"
                tasks.at[index, "failure_reason"] = pd.NA
                tasks.at[index, "attempt_count"] = 0
                tasks.at[index, "started_at"] = pd.NA
                tasks.at[index, "finished_at"] = pd.NA
                tasks.at[index, "runtime_seconds"] = pd.NA
                tasks.at[index, "result_subdir"] = "."
            elif method not in enabled_methods:
                tasks.at[index, "status"] = "skipped_with_reason"
                tasks.at[index, "failure_reason"] = registered["notes"]
            method_root = replicate["methods"] / method
            if method_root.is_dir() and str(tasks.at[index, "status"]) != "pending":
                for csv_name in ("public_contrast.csv", "evidence.csv", "diagnostics.csv"):
                    csv_path = method_root / csv_name
                    if csv_path.is_file() and csv_path.stat().st_size > 1:
                        try:
                            frame = pd.read_csv(csv_path)
                        except pd.errors.EmptyDataError:
                            continue
                        if "run_id" in frame:
                            frame["run_id"] = new_run_id
                            frame.to_csv(csv_path, index=False)
                task_manifest_path = method_root / "task_manifest.json"
                if task_manifest_path.is_file():
                    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
                    task_manifest["benchmark_id"] = benchmark_id
                    task_manifest["run_id"] = new_run_id
                    task_manifest["files"] = _file_records(
                        method_root, excluded={"task_manifest.json"}
                    )
                    _write_json(task_manifest_path, task_manifest)
        run_manifest["files"] = _file_records(
            replicate["root"], excluded={"manifests/run_manifest.json"}
        )
        _write_json(run_manifest_path, run_manifest)

    _write_task_manifest(tasks, target / "benchmark_task_manifest.csv")
    derived_manifest = {
        "schema_version": "phase6-expanded-benchmark-run-v1",
        "benchmark_id": benchmark_id,
        "phase": source_manifest["phase"],
        "replicates_per_scenario": source_manifest["replicates_per_scenario"],
        "scenario_ids": source_manifest["scenario_ids"],
        "enabled_methods": sorted(enabled_methods),
        "status": "derived_inputs_and_existing_outputs_frozen_new_methods_pending",
        "created_at": _utc_now(),
        "derived_from_benchmark_id": source.name,
        "source_benchmark_status": source_manifest["status"],
        "frozen_input_policy": "byte_identical_canonical_inputs_reused",
        "files": _file_records(
            target, excluded={"benchmark_manifest.json", "benchmark_task_manifest.csv"}
        ),
    }
    _write_json(target / "benchmark_manifest.json", derived_manifest)
    return target


def _load_frozen_canonical(replicate_root: Path) -> CanonicalDAInput:
    simulation = replicate_root / "simulation"
    canonical = CanonicalDAInput(
        pd.read_csv(simulation / "canonical_abundance_input.csv"),
        pd.read_csv(simulation / "sample_metadata.csv"),
        pd.read_csv(simulation / "cell_type_manifest.csv"),
        pd.read_csv(simulation / "contrast_specification.csv"),
    ).validate()
    manifest = json.loads(
        (replicate_root / "manifests" / "run_manifest.json").read_text(encoding="utf-8")
    )
    if canonical.input_hash() != manifest["input_hash"]:
        raise ValueError("Frozen canonical input hash does not match its replicate manifest.")
    return canonical


def _save_native_input(adapter: Any, canonical: CanonicalDAInput, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    contrast = canonical.contrast_specification.iloc[0]
    native = adapter.prepare_native_input(canonical, contrast)
    native.abundance.to_csv(target / "abundance.csv", index=False)
    native.sample_manifest.to_csv(target / "sample_manifest.csv", index=False)
    native.cell_type_manifest.to_csv(target / "cell_type_manifest.csv", index=False)
    _write_json(target / "run_spec.json", {
        "contrast": native.contrast,
        "options": native.options,
    })


def _environment_record(method: str) -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "anndata", "pertpy", "jax", "numpyro", "arviz"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "method": method,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }


def _memory_rss() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def run_pending_tasks(
    benchmark_root: str | Path,
    *,
    only_methods: set[str] | None = None,
    exclude_methods: set[str] | None = None,
    only_replicates: set[str] | None = None,
) -> Path:
    root = Path(benchmark_root).resolve()
    task_path = root / "benchmark_task_manifest.csv"
    tasks = pd.read_csv(task_path)
    for column in ("failure_reason", "started_at", "finished_at", "result_subdir"):
        if column not in tasks:
            tasks[column] = "." if column == "result_subdir" else pd.NA
        tasks[column] = tasks[column].astype("object")
    config = yaml.safe_load(_run_config_path(root).read_text(encoding="utf-8"))
    order = list(config["execution"]["enabled_task_order"])
    order_rank = {method: index for index, method in enumerate(order)}
    pending_mask = tasks["status"].eq("pending")
    if only_methods:
        pending_mask &= tasks["method"].astype(str).isin(only_methods)
    if exclude_methods:
        pending_mask &= ~tasks["method"].astype(str).isin(exclude_methods)
    if only_replicates:
        pending_mask &= tasks["replicate_id"].astype(str).isin(only_replicates)
    pending_indices = tasks.index[pending_mask].tolist()
    pending_indices.sort(key=lambda index: (
        order_rank.get(str(tasks.at[index, "method"]), len(order_rank)),
        str(tasks.at[index, "scenario_id"]),
        str(tasks.at[index, "replicate_id"]),
    ))
    for index in pending_indices:
        task = tasks.loc[index]
        method = str(task["method"])
        replicate = _replicate_paths(root, str(task["scenario_id"]), str(task["replicate_id"]))
        method_root = replicate["methods"] / method
        if method_root.exists():
            if any(method_root.iterdir()) or int(task["attempt_count"]) != 0:
                raise FileExistsError(f"Pending task output already exists: {method_root}")
        else:
            method_root.mkdir()
        tasks.at[index, "status"] = "running"
        tasks.at[index, "attempt_count"] = int(task["attempt_count"]) + 1
        tasks.at[index, "started_at"] = _utc_now()
        _write_task_manifest(tasks, task_path)
        started = time.perf_counter()
        memory_before = _memory_rss()
        final_status = "conversion_failed"
        failure_reason: str | None = None
        try:
            canonical = _load_frozen_canonical(replicate["root"])
            if canonical.input_hash() != str(task["input_hash"]):
                raise ValueError("Task input hash disagrees with frozen replicate input.")
            runtime = dict(config.get("runtime", {}))
            runtime["sccoda_rng_key"] = int(task["simulation_seed"])
            runtime["engine_rng_seed"] = int(task["simulation_seed"])
            adapter_config = {
                "methods": [method],
                "contrast": canonical.contrast_specification.iloc[0].to_dict(),
                "runtime": runtime,
            }
            adapter = _build_adapters(adapter_config, method_root)[0]
            _save_native_input(adapter, canonical, method_root / "native_input")
            _write_json(method_root / "environment.json", _environment_record(method))
            result = DifferentialAbundanceRunner(
                method_root / "runner_outputs", _decision_registry([method])
            ).run(
                canonical,
                [adapter],
                analysis_id=str(task["analysis_id"]),
                run_id=str(task["run_id"]),
            )
            result.public_view.to_csv(method_root / "public_contrast.csv", index=False)
            result.evidence_layer.to_csv(method_root / "evidence.csv", index=False)
            result.diagnostics.to_csv(method_root / "diagnostics.csv", index=False)
            diagnostic = result.diagnostics.iloc[0]
            details = diagnostic.get("details", {})
            if not isinstance(details, dict):
                details = {}
            (method_root / "stdout.txt").write_text(str(details.get("stdout", "")), encoding="utf-8")
            (method_root / "stderr.txt").write_text(str(details.get("stderr", "")), encoding="utf-8")
            diagnostic_status = str(diagnostic["status"])
            if diagnostic_status == "success" and bool(diagnostic.get("converged", False)):
                final_status = "success"
            elif diagnostic_status == "failed":
                final_status = "runtime_failed"
                failure_reason = str(diagnostic.get("error_message") or diagnostic_status)
            elif diagnostic_status == "diagnostics_invalid" or not bool(diagnostic.get("converged", False)):
                final_status = "diagnostics_invalid"
                failure_reason = str(diagnostic.get("error_message") or "native_diagnostics_invalid")
            else:
                final_status = "runtime_failed"
                failure_reason = str(diagnostic.get("error_message") or diagnostic_status)
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            (method_root / "stderr.txt").write_text(failure_reason, encoding="utf-8")
            if not (method_root / "stdout.txt").exists():
                (method_root / "stdout.txt").write_text("", encoding="utf-8")
        runtime_seconds = time.perf_counter() - started
        memory_after = _memory_rss()
        task_manifest = {
            "benchmark_id": task["benchmark_id"],
            "scenario_id": task["scenario_id"],
            "replicate_id": task["replicate_id"],
            "run_id": task["run_id"],
            "analysis_id": task["analysis_id"],
            "method": method,
            "benchmark_role": task["benchmark_role"],
            "scientific_status": task["scientific_status"],
            "status": final_status,
            "failure_reason": failure_reason,
            "input_hash": task["input_hash"],
            "runtime_seconds": runtime_seconds,
            "memory_rss_before_bytes": memory_before,
            "memory_rss_after_bytes": memory_after,
            "memory_measurement": "parent_process_rss_before_after; child_peak_unavailable",
            "attempt_count": int(tasks.at[index, "attempt_count"]),
            "files": _file_records(method_root, excluded={"task_manifest.json"}),
        }
        _write_json(method_root / "task_manifest.json", task_manifest)
        tasks.at[index, "status"] = final_status
        tasks.at[index, "failure_reason"] = failure_reason
        tasks.at[index, "finished_at"] = _utc_now()
        tasks.at[index, "runtime_seconds"] = runtime_seconds
        _write_task_manifest(tasks, task_path)
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "method_tasks_complete",
        "methods_finished_at": _utc_now(),
        "task_status_counts": tasks["status"].value_counts().to_dict(),
    })
    _write_json(manifest_path, manifest)
    return task_path


def _execute_parallel_task(
    root_value: str, task_record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Execute one isolated method/replicate task; the parent owns the CSV manifest."""
    root = Path(root_value)
    task = pd.Series(task_record)
    method = str(task["method"])
    replicate = _replicate_paths(root, str(task["scenario_id"]), str(task["replicate_id"]))
    method_root = replicate["methods"] / method
    method_root.mkdir(exist_ok=False)
    started = time.perf_counter()
    memory_before = _memory_rss()
    final_status = "conversion_failed"
    failure_reason: str | None = None
    try:
        canonical = _load_frozen_canonical(replicate["root"])
        if canonical.input_hash() != str(task["input_hash"]):
            raise ValueError("Task input hash disagrees with frozen replicate input.")
        runtime = dict(config.get("runtime", {}))
        runtime["sccoda_rng_key"] = int(task["simulation_seed"])
        runtime["engine_rng_seed"] = int(task["simulation_seed"])
        # Replicate-level processes provide the parallelism; avoiding nested
        # cell-type pools prevents oversubscription and keeps each worker isolated.
        runtime["engine_max_workers"] = 1
        adapter = _build_adapters({
            "methods": [method],
            "contrast": canonical.contrast_specification.iloc[0].to_dict(),
            "runtime": runtime,
        }, method_root)[0]
        _save_native_input(adapter, canonical, method_root / "native_input")
        _write_json(method_root / "environment.json", _environment_record(method))
        result = DifferentialAbundanceRunner(
            method_root / "runner_outputs", _decision_registry([method])
        ).run(
            canonical, [adapter], analysis_id=str(task["analysis_id"]),
            run_id=str(task["run_id"]),
        )
        result.public_view.to_csv(method_root / "public_contrast.csv", index=False)
        result.evidence_layer.to_csv(method_root / "evidence.csv", index=False)
        result.diagnostics.to_csv(method_root / "diagnostics.csv", index=False)
        diagnostic = result.diagnostics.iloc[0]
        details = diagnostic.get("details", {})
        if not isinstance(details, dict):
            details = {}
        (method_root / "stdout.txt").write_text(str(details.get("stdout", "")), encoding="utf-8")
        (method_root / "stderr.txt").write_text(str(details.get("stderr", "")), encoding="utf-8")
        diagnostic_status = str(diagnostic["status"])
        if diagnostic_status == "success" and bool(diagnostic.get("converged", False)):
            final_status = "success"
        elif diagnostic_status == "failed":
            final_status = "runtime_failed"
            failure_reason = str(diagnostic.get("error_message") or diagnostic_status)
        elif diagnostic_status == "diagnostics_invalid" or not bool(diagnostic.get("converged", False)):
            final_status = "diagnostics_invalid"
            failure_reason = str(diagnostic.get("error_message") or "native_diagnostics_invalid")
        else:
            final_status = "runtime_failed"
            failure_reason = str(diagnostic.get("error_message") or diagnostic_status)
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        (method_root / "stderr.txt").write_text(failure_reason, encoding="utf-8")
        if not (method_root / "stdout.txt").exists():
            (method_root / "stdout.txt").write_text("", encoding="utf-8")
    runtime_seconds = time.perf_counter() - started
    task_manifest = {
        "benchmark_id": task["benchmark_id"], "scenario_id": task["scenario_id"],
        "replicate_id": task["replicate_id"], "run_id": task["run_id"],
        "analysis_id": task["analysis_id"], "method": method,
        "benchmark_role": task["benchmark_role"],
        "scientific_status": task["scientific_status"], "status": final_status,
        "failure_reason": failure_reason, "input_hash": task["input_hash"],
        "runtime_seconds": runtime_seconds, "memory_rss_before_bytes": memory_before,
        "memory_rss_after_bytes": _memory_rss(),
        "memory_measurement": "worker_process_rss_before_after; child_peak_unavailable",
        "attempt_count": int(task["attempt_count"]),
        "execution_scheduler": "replicate_level_process_pool",
        "files": _file_records(method_root, excluded={"task_manifest.json"}),
    }
    _write_json(method_root / "task_manifest.json", task_manifest)
    return {
        "index": int(task["_manifest_index"]), "status": final_status,
        "failure_reason": failure_reason, "runtime_seconds": runtime_seconds,
        "finished_at": _utc_now(),
    }


def run_pending_tasks_parallel(
    benchmark_root: str | Path, *, max_workers: int = 4,
    only_methods: set[str] | None = None,
) -> Path:
    """Run independent replicate tasks in processes while serializing manifest writes."""
    if max_workers < 2:
        return run_pending_tasks(benchmark_root, only_methods=only_methods)
    root = Path(benchmark_root).resolve()
    task_path = root / "benchmark_task_manifest.csv"
    tasks = pd.read_csv(task_path)
    for column in ("failure_reason", "started_at", "finished_at", "result_subdir"):
        if column not in tasks:
            tasks[column] = "." if column == "result_subdir" else pd.NA
        tasks[column] = tasks[column].astype("object")
    config = yaml.safe_load(_run_config_path(root).read_text(encoding="utf-8"))
    order = list(config["execution"]["enabled_task_order"])
    selected = tasks["status"].eq("pending")
    if only_methods:
        selected &= tasks["method"].astype(str).isin(only_methods)
    for method in order:
        indices = tasks.index[selected & tasks["method"].astype(str).eq(method)].tolist()
        if not indices:
            continue
        for index in indices:
            method_root = _replicate_paths(
                root, str(tasks.at[index, "scenario_id"]), str(tasks.at[index, "replicate_id"])
            )["methods"] / method
            if method_root.exists():
                raise FileExistsError(f"Pending task output already exists: {method_root}")
            tasks.at[index, "status"] = "running"
            tasks.at[index, "attempt_count"] = int(tasks.at[index, "attempt_count"]) + 1
            tasks.at[index, "started_at"] = _utc_now()
        _write_task_manifest(tasks, task_path)
        records = []
        for index in indices:
            record = tasks.loc[index].to_dict()
            record["_manifest_index"] = int(index)
            records.append(record)
        method_workers = 1 if method == "pydeseq2" else min(max_workers, len(records))
        with ProcessPoolExecutor(max_workers=method_workers) as pool:
            futures = [
                pool.submit(_execute_parallel_task, str(root), record, config)
                for record in records
            ]
            for future in as_completed(futures):
                outcome = future.result()
                index = outcome.pop("index")
                for key, value in outcome.items():
                    tasks.at[index, key] = value
                tasks.at[index, "result_subdir"] = "."
                _write_task_manifest(tasks, task_path)
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "method_tasks_complete",
        "methods_finished_at": _utc_now(),
        "task_status_counts": tasks["status"].value_counts().to_dict(),
        "execution_scheduler": {
            "type": "replicate_level_process_pool", "max_workers": max_workers,
            "nested_engine_workers": 1,
        },
    })
    _write_json(manifest_path, manifest)
    return task_path


def production_checkpoint(
    benchmark_root: str | Path,
    *,
    replicate_ids: set[str],
) -> Path:
    """Evaluate the registered 10%-20% production safety gate without finalizing the run."""
    root = Path(benchmark_root).resolve()
    tasks = pd.read_csv(root / "benchmark_task_manifest.csv")
    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("phase") != "production":
        raise ValueError("The production checkpoint only applies to production benchmarks.")
    selected = tasks.loc[tasks["replicate_id"].astype(str).isin(replicate_ids)].copy()
    enabled = selected.loc[selected["status"].ne("skipped_with_reason")].copy()
    if enabled.empty or enabled["status"].isin(["pending", "running"]).any():
        raise RuntimeError("Every enabled checkpoint task must be terminal before the safety gate.")
    registry = load_method_registry(root / "method_registry" / "method_benchmark_registry.yaml")
    scenarios = load_scenario_matrix(root / "benchmark_config" / "scenario_matrix.csv")
    public_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    hash_failures: list[dict[str, str]] = []
    for (scenario_id, replicate_id), group in selected.groupby(
        ["scenario_id", "replicate_id"], sort=False
    ):
        paths = _replicate_paths(root, str(scenario_id), str(replicate_id))
        truth_frames.append(pd.read_csv(paths["truth"] / "truth_table.csv"))
        active = group.loc[group["status"].ne("skipped_with_reason")]
        if active["input_hash"].astype(str).nunique() != 1:
            hash_failures.append({
                "scenario_id": str(scenario_id),
                "replicate_id": str(replicate_id),
                "reason": "method_input_hash_mismatch",
            })
        frozen = json.loads(
            (paths["manifests"] / "run_manifest.json").read_text(encoding="utf-8")
        )["input_hash"]
        if not active["input_hash"].astype(str).eq(str(frozen)).all():
            hash_failures.append({
                "scenario_id": str(scenario_id),
                "replicate_id": str(replicate_id),
                "reason": "task_hash_disagrees_with_frozen_manifest",
            })
        for _, task in active.iterrows():
            result_root = paths["methods"] / str(task["method"]) / str(task.get("result_subdir", "."))
            public_path = result_root / "public_contrast.csv"
            if public_path.is_file():
                public_frames.append(pd.read_csv(public_path))
    truth = validate_truth_table(pd.concat(truth_frames, ignore_index=True))
    public = pd.concat(public_frames, ignore_index=True)
    enabled_methods = tuple(registry.loc[registry["enabled"].astype(bool), "method_id"].astype(str))
    evaluation = evaluate_contrasts(
        public,
        truth,
        EvaluationSpec(methods=enabled_methods),
    )
    metric_columns = [
        "Power", "FPR", "Specificity", "Precision", "FDP_descriptive", "FDP_for_FDR",
    ]
    metric_error = False
    for column in metric_columns:
        values = pd.to_numeric(evaluation.replicate_metrics[column], errors="coerce").dropna()
        if (~values.between(0, 1)).any():
            metric_error = True
    population = truth.loc[truth["truth_source"].eq("population")].copy()
    population_flags = parse_boolean_series(population["is_true_effect"])
    population["is_true_effect"] = population_flags
    truth_status = (
        population.groupby(["scenario_id", "replicate_id"])["is_true_effect"]
        .agg(number_true=lambda values: int(values.eq(True).sum()),
             number_null=lambda values: int(values.eq(False).sum()))
        .reset_index()
    )
    scenario_effects = scenarios.set_index("scenario_id")["effect_strength"].to_dict()
    truth_invalid = False
    for _, row in truth_status.iterrows():
        if scenario_effects.get(row["scenario_id"]) == "null":
            truth_invalid |= int(row["number_true"]) != 0 or int(row["number_null"]) == 0
        else:
            truth_invalid |= int(row["number_true"]) == 0 or int(row["number_null"]) == 0
    status = (
        enabled.groupby(["method", "status"]).size().reset_index(name="number_tasks")
    )
    per_method = enabled.groupby("method")["status"].agg(
        number_tasks="size",
        number_success=lambda values: int(values.eq("success").sum()),
        number_diagnostics_invalid=lambda values: int(values.eq("diagnostics_invalid").sum()),
        number_runtime_failed=lambda values: int(values.eq("runtime_failed").sum()),
        number_conversion_failed=lambda values: int(values.eq("conversion_failed").sum()),
    ).reset_index()
    systematic_missing = bool(per_method["number_success"].eq(0).sum() >= 2)
    all_files = [path for path in root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in all_files)
    completed_fraction = len(replicate_ids) / int(manifest["replicates_per_scenario"])
    method_bytes = sum(
        path.stat().st_size for path in all_files if "methods" in path.relative_to(root).parts
    )
    static_bytes = total_bytes - method_bytes
    projected_bytes = (
        int(static_bytes + method_bytes / completed_fraction)
        if completed_fraction else 0
    )
    free_bytes = shutil.disk_usage(root).free
    resources_exceeded = projected_bytes > free_bytes * 0.8
    blockers = []
    if hash_failures:
        blockers.append("method_input_hash_mismatch")
    if truth_invalid:
        blockers.append("truth_distribution_invalid")
    if metric_error:
        blockers.append("metric_out_of_bounds")
    if systematic_missing:
        blockers.append("multiple_methods_systematically_missing")
    if resources_exceeded:
        blockers.append("projected_storage_exceeds_plan")
    checkpoint_percent = int(round(100 * completed_fraction))
    checkpoint_name = f"{checkpoint_percent}_percent"
    checkpoint_root = root / "diagnostics" / f"production_checkpoint_{checkpoint_percent}pct"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    status.to_csv(checkpoint_root / "task_status.csv", index=False)
    per_method.to_csv(checkpoint_root / "method_status.csv", index=False)
    truth_status.to_csv(checkpoint_root / "truth_distribution.csv", index=False)
    evaluation.replicate_metrics.to_csv(checkpoint_root / "replicate_metrics.csv", index=False)
    pd.DataFrame(hash_failures, columns=["scenario_id", "replicate_id", "reason"]).to_csv(
        checkpoint_root / "input_hash_failures.csv", index=False
    )
    report = {
        "checkpoint": checkpoint_name,
        "replicate_ids": sorted(replicate_ids),
        "number_selected_tasks": len(enabled),
        "input_hash_check": "passed" if not hash_failures else "failed",
        "truth_distribution_check": "passed" if not truth_invalid else "failed",
        "metric_bounds_check": "passed" if not metric_error else "failed",
        "systematic_missing_check": "passed" if not systematic_missing else "failed",
        "actual_bytes": total_bytes,
        "projected_bytes": projected_bytes,
        "free_bytes": free_bytes,
        "resource_check": "passed" if not resources_exceeded else "failed",
        "blockers": blockers,
        "status": "self_accepted" if not blockers else "blocked",
        "created_at": _utc_now(),
        "files": _file_records(checkpoint_root, excluded={"checkpoint_manifest.json"}),
    }
    _write_json(checkpoint_root / "checkpoint_manifest.json", report)
    _write_json(checkpoint_root / "checkpoint_summary.json", report)
    pd.DataFrame({"blocker": blockers}).to_csv(
        checkpoint_root / "checkpoint_failures.csv", index=False
    )
    if blockers:
        raise RuntimeError(f"Production checkpoint blockers: {blockers}")
    checkpoint_record = {
        "status": "self_accepted",
        "checkpoint": checkpoint_name,
        "replicate_ids": sorted(replicate_ids),
        "created_at": report["created_at"],
    }
    if _phase_label_for_run(root) == "phase5":
        manifest["stop_point_c"] = checkpoint_record
    else:
        existing = list(manifest.get("production_checkpoints", []))
        existing.append(checkpoint_record)
        manifest["production_checkpoints"] = existing
    _write_json(root / "benchmark_manifest.json", manifest)
    return checkpoint_root / "checkpoint_manifest.json"


def retry_failed_tasks(
    benchmark_root: str | Path,
    *,
    method: str,
    reason_contains: str,
) -> Path:
    """Run one explicit technical retry while retaining the complete first attempt."""
    root = Path(benchmark_root).resolve()
    task_path = root / "benchmark_task_manifest.csv"
    tasks = pd.read_csv(task_path)
    for column in ("failure_reason", "started_at", "finished_at", "result_subdir"):
        if column not in tasks:
            tasks[column] = "." if column == "result_subdir" else pd.NA
        tasks[column] = tasks[column].astype("object")
    config = yaml.safe_load(
        _run_config_path(root).read_text(encoding="utf-8")
    )
    candidates = tasks.index[
        tasks["method"].astype(str).eq(method)
        & tasks["status"].isin(
            ["runtime_failed", "diagnostics_invalid", "conversion_failed"]
        )
        & pd.to_numeric(tasks["attempt_count"], errors="coerce").eq(1)
        & tasks["failure_reason"].astype(str).str.contains(reason_contains, case=False, regex=False)
    ].tolist()
    if not candidates:
        raise ValueError("No eligible first-attempt failure matches the requested technical retry.")
    for index in candidates:
        task = tasks.loc[index]
        replicate = _replicate_paths(root, str(task["scenario_id"]), str(task["replicate_id"]))
        method_root = replicate["methods"] / method
        original_manifest_path = method_root / "task_manifest.json"
        original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
        attempt_root = method_root / "attempts" / "attempt-002"
        attempt_root.mkdir(parents=True, exist_ok=False)
        tasks.at[index, "status"] = "running"
        tasks.at[index, "attempt_count"] = 2
        tasks.at[index, "started_at"] = _utc_now()
        _write_task_manifest(tasks, task_path)
        started = time.perf_counter()
        memory_before = _memory_rss()
        final_status = "conversion_failed"
        failure_reason: str | None = None
        try:
            canonical = _load_frozen_canonical(replicate["root"])
            if canonical.input_hash() != str(task["input_hash"]):
                raise ValueError("Retry input hash disagrees with the frozen first-attempt input.")
            runtime = dict(config.get("runtime", {}))
            runtime["sccoda_rng_key"] = int(task["simulation_seed"])
            runtime["engine_rng_seed"] = int(task["simulation_seed"])
            adapter = _build_adapters({
                "methods": [method],
                "contrast": canonical.contrast_specification.iloc[0].to_dict(),
                "runtime": runtime,
            }, attempt_root)[0]
            _save_native_input(adapter, canonical, attempt_root / "native_input")
            _write_json(attempt_root / "environment.json", _environment_record(method))
            result = DifferentialAbundanceRunner(
                attempt_root / "runner_outputs", _decision_registry([method])
            ).run(
                canonical,
                [adapter],
                analysis_id=str(task["analysis_id"]),
                run_id=str(task["run_id"]),
            )
            result.public_view.to_csv(attempt_root / "public_contrast.csv", index=False)
            result.evidence_layer.to_csv(attempt_root / "evidence.csv", index=False)
            result.diagnostics.to_csv(attempt_root / "diagnostics.csv", index=False)
            diagnostic = result.diagnostics.iloc[0]
            details = diagnostic.get("details", {})
            if not isinstance(details, dict):
                details = {}
            (attempt_root / "stdout.txt").write_text(str(details.get("stdout", "")), encoding="utf-8")
            (attempt_root / "stderr.txt").write_text(str(details.get("stderr", "")), encoding="utf-8")
            diagnostic_status = str(diagnostic["status"])
            if diagnostic_status == "success" and bool(diagnostic.get("converged", False)):
                final_status = "success"
            elif diagnostic_status == "failed":
                final_status = "runtime_failed"
                failure_reason = str(diagnostic.get("error_message") or diagnostic_status)
            else:
                final_status = "diagnostics_invalid"
                failure_reason = str(diagnostic.get("error_message") or "native_diagnostics_invalid")
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            (attempt_root / "stderr.txt").write_text(failure_reason, encoding="utf-8")
            if not (attempt_root / "stdout.txt").exists():
                (attempt_root / "stdout.txt").write_text("", encoding="utf-8")
        runtime_seconds = time.perf_counter() - started
        retry_manifest = {
            "benchmark_id": task["benchmark_id"],
            "scenario_id": task["scenario_id"],
            "replicate_id": task["replicate_id"],
            "run_id": task["run_id"],
            "analysis_id": task["analysis_id"],
            "method": method,
            "status": final_status,
            "failure_reason": failure_reason,
            "input_hash": task["input_hash"],
            "runtime_seconds": runtime_seconds,
            "memory_rss_before_bytes": memory_before,
            "memory_rss_after_bytes": _memory_rss(),
            "memory_measurement": "parent_process_rss_before_after; child_peak_unavailable",
            "attempt_count": 2,
            "retry_reason": (
                "Explicit technical retry after the recorded first-attempt failure matched: "
                f"{reason_contains}"
            ),
            "files": _file_records(attempt_root, excluded={"task_manifest.json"}),
        }
        _write_json(attempt_root / "task_manifest.json", retry_manifest)
        _write_json(method_root / "retry_history.json", {
            "automatic_retry": False,
            "maximum_attempts": 2,
            "attempts": [
                {
                    "attempt": 1,
                    "path": ".",
                    "recorded_status": original_manifest["status"],
                    "corrected_failure_class": original_manifest["status"],
                    "failure_reason": original_manifest.get("failure_reason"),
                    "task_manifest": "task_manifest.json",
                },
                {
                    "attempt": 2,
                    "path": "attempts/attempt-002",
                    "status": final_status,
                    "failure_reason": failure_reason,
                    "task_manifest": "attempts/attempt-002/task_manifest.json",
                },
            ],
        })
        tasks.at[index, "status"] = final_status
        tasks.at[index, "failure_reason"] = failure_reason
        tasks.at[index, "finished_at"] = _utc_now()
        tasks.at[index, "runtime_seconds"] = runtime_seconds
        tasks.at[index, "result_subdir"] = "attempts/attempt-002"
        _write_task_manifest(tasks, task_path)
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "method_tasks_complete_after_recorded_retry",
        "technical_retry": {
            "method": method,
            "reason_contains": reason_contains,
            "automatic": False,
            "number_tasks": len(candidates),
            "finished_at": _utc_now(),
        },
        "task_status_counts": tasks["status"].value_counts().to_dict(),
    })
    _write_json(manifest_path, manifest)
    return task_path


def migrate_dirichlet_estimand_contract(benchmark_root: str | Path) -> Path:
    """Refresh only the deterministic public estimand classification for Dirichlet outputs.

    The statistical estimates, evidence, decisions, diagnostics, and task status are left
    unchanged.  The migration is deliberately narrow and records every before/after hash.
    """
    root = Path(benchmark_root).resolve()
    task_path = root / "benchmark_task_manifest.csv"
    tasks = pd.read_csv(task_path)
    methods = {"dirichlet_wald", "dirichlet_multinomial_wald"}
    selected = tasks.loc[
        tasks["method"].astype(str).isin(methods)
        & tasks["status"].astype(str).eq("success")
    ].copy()
    if len(selected) != 280:
        raise ValueError(
            "The Phase-6 Dirichlet contract migration requires exactly 280 successful tasks."
        )
    records: list[dict[str, Any]] = []
    for task in selected.itertuples(index=False):
        replicate = _replicate_paths(root, str(task.scenario_id), str(task.replicate_id))
        result_subdir = str(getattr(task, "result_subdir", ".") or ".")
        result_root = replicate["methods"] / str(task.method) / result_subdir
        public_path = result_root / "public_contrast.csv"
        manifest_path = result_root / "task_manifest.json"
        before_hash = _sha256(public_path)
        public = pd.read_csv(public_path)
        required = {
            "effect_estimand", "effect_scale", "reference_cell_type",
            "estimand_compatibility",
        }
        if missing := required - set(public.columns):
            raise ValueError(f"Dirichlet public output lacks contract fields: {sorted(missing)}")
        assessable_mask = public["estimand_compatibility"].astype(str).ne("unavailable")
        expected_estimand = {
            "dirichlet_wald": "dirichlet_log_alpha_contrast",
            "dirichlet_multinomial_wald": "dirichlet_multinomial_log_alpha_contrast",
        }[str(task.method)]
        if not public.loc[assessable_mask, "effect_estimand"].astype(str).eq(
            expected_estimand
        ).all():
            raise ValueError("Migration encountered an unexpected Dirichlet estimand.")
        if not public.loc[assessable_mask, "effect_scale"].astype(str).eq("log_ratio").all():
            raise ValueError("Migration encountered an unexpected Dirichlet effect scale.")
        if public.loc[assessable_mask, "reference_cell_type"].isna().any() or public.loc[
            assessable_mask, "reference_cell_type"
        ].astype(str).eq("").any():
            raise ValueError("Dirichlet reference cell type must be explicit for migration.")
        existing = set(public["estimand_compatibility"].astype(str))
        if not existing.issubset({"incompatible", "direction_only", "unavailable"}):
            raise ValueError(f"Unexpected pre-migration compatibility values: {sorted(existing)}")
        migration_mask = public["estimand_compatibility"].astype(str).eq("incompatible")
        changed_rows = int(migration_mask.sum())
        public.loc[migration_mask, "estimand_compatibility"] = "direction_only"
        public.to_csv(public_path, index=False)
        after_hash = _sha256(public_path)
        task_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        task_manifest["files"] = _file_records(
            result_root, excluded={"task_manifest.json"}
        )
        task_manifest["public_contract_migration"] = {
            "migration_id": "dirichlet-reference-estimand-direction-v1",
            "changed_rows": changed_rows,
            "statistical_values_changed": False,
        }
        _write_json(manifest_path, task_manifest)
        records.append({
            "scenario_id": task.scenario_id,
            "replicate_id": task.replicate_id,
            "method": task.method,
            "result_subdir": result_subdir,
            "changed_rows": changed_rows,
            "public_sha256_before": before_hash,
            "public_sha256_after": after_hash,
        })
    migration_root = root / "contract_migrations"
    migration_root.mkdir(parents=True, exist_ok=True)
    record_path = migration_root / "dirichlet-reference-estimand-direction-v1.csv"
    pd.DataFrame(records).to_csv(record_path, index=False)
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    migrations = list(manifest.get("contract_migrations", []))
    migrations.append({
        "migration_id": "dirichlet-reference-estimand-direction-v1",
        "created_at": _utc_now(),
        "number_tasks": len(records),
        "number_rows_changed": int(sum(row["changed_rows"] for row in records)),
        "statistical_values_changed": False,
        "record": record_path.relative_to(root).as_posix(),
    })
    manifest["contract_migrations"] = migrations
    _write_json(manifest_path, manifest)
    return record_path


def accept_phase6_stop_point_c(benchmark_root: str | Path) -> Path:
    """Write and validate the all-method completion package before evaluation."""
    root = Path(benchmark_root).resolve()
    if _phase_label_for_run(root) != "phase6":
        raise ValueError("This Stop Point C gate is specific to Phase 6.")
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "production":
        raise ValueError("Phase-6 Stop Point C applies only to the production run.")
    tasks = pd.read_csv(root / "benchmark_task_manifest.csv")
    enabled = tasks.loc[tasks["status"].ne("skipped_with_reason")].copy()
    completion = (
        enabled.groupby(["method", "status"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for status in sorted(ALLOWED_TASK_STATUSES - {"skipped_with_reason"}):
        if status not in completion:
            completion[status] = 0
    completion["number_tasks"] = completion[
        sorted(ALLOWED_TASK_STATUSES - {"skipped_with_reason"})
    ].sum(axis=1)
    completion["terminal_rate"] = 1 - (
        completion["pending"] + completion["running"]
    ) / completion["number_tasks"]
    completion.to_csv(root / "method_completion_matrix.csv", index=False)

    failures = enabled.loc[~enabled["status"].eq("success")].copy()
    failures.to_csv(root / "method_failure_summary.csv", index=False)
    enabled[[
        "scenario_id", "replicate_id", "method", "status", "runtime_seconds",
        "attempt_count", "failure_reason", "result_subdir",
    ]].to_csv(root / "runtime_summary.csv", index=False)

    diagnostics_frames: list[pd.DataFrame] = []
    missing_outputs: list[dict[str, str]] = []
    hash_failures: list[dict[str, str]] = []
    for _, task in enabled.iterrows():
        paths = _replicate_paths(root, str(task["scenario_id"]), str(task["replicate_id"]))
        frozen_hash = json.loads(
            (paths["manifests"] / "run_manifest.json").read_text(encoding="utf-8")
        )["input_hash"]
        if str(task["input_hash"]) != str(frozen_hash):
            hash_failures.append({
                "scenario_id": str(task["scenario_id"]),
                "replicate_id": str(task["replicate_id"]),
                "method": str(task["method"]),
                "reason": "task_hash_disagrees_with_frozen_manifest",
            })
        if str(task["status"]) != "success":
            continue
        result_root = (
            paths["methods"] / str(task["method"]) / str(task.get("result_subdir", "."))
        )
        required_outputs = ("public_contrast.csv", "evidence.csv", "diagnostics.csv")
        absent = [name for name in required_outputs if not (result_root / name).is_file()]
        if absent:
            missing_outputs.append({
                "scenario_id": str(task["scenario_id"]),
                "replicate_id": str(task["replicate_id"]),
                "method": str(task["method"]),
                "reason": f"missing_success_outputs:{','.join(absent)}",
            })
            continue
        diagnostics_frames.append(pd.read_csv(result_root / "diagnostics.csv"))
    diagnostics = (
        pd.concat(diagnostics_frames, ignore_index=True)
        if diagnostics_frames else pd.DataFrame()
    )
    diagnostics.to_csv(root / "diagnostic_summary.csv", index=False)

    blockers: list[str] = []
    if enabled["status"].isin(["pending", "running"]).any():
        blockers.append("enabled_tasks_not_terminal")
    per_method_success = enabled.groupby("method")["status"].apply(
        lambda values: int(values.eq("success").sum())
    )
    if per_method_success.eq(0).any():
        blockers.append("enabled_method_has_no_successful_run")
    if missing_outputs:
        blockers.append("successful_task_missing_canonical_outputs")
    if hash_failures:
        blockers.append("input_hash_mismatch")
    readiness = root / "evaluation_readiness_report.md"
    lines = [
        "# Phase 6 evaluation readiness",
        "",
        f"- Enabled tasks: {len(enabled)}",
        f"- Successful tasks: {int(enabled['status'].eq('success').sum())}",
        f"- Terminal tasks: {int((~enabled['status'].isin(['pending', 'running'])).sum())}",
        f"- Enabled methods with at least one success: {int(per_method_success.gt(0).sum())}",
        f"- Hash mismatches: {len(hash_failures)}",
        f"- Successful tasks missing outputs: {len(missing_outputs)}",
        f"- Blockers: {', '.join(blockers) if blockers else 'none'}",
        "",
        "Stop Point C: self-accepted" if not blockers else "Stop Point C: blocked",
        "",
    ]
    readiness.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(missing_outputs).to_csv(root / "missing_success_outputs.csv", index=False)
    pd.DataFrame(hash_failures).to_csv(root / "completion_hash_failures.csv", index=False)
    if blockers:
        raise RuntimeError(f"Phase-6 Stop Point C blockers: {blockers}")
    manifest["stop_point_c"] = {
        "status": "self_accepted",
        "created_at": _utc_now(),
        "evaluation_readiness_report": "evaluation_readiness_report.md",
    }
    _write_json(manifest_path, manifest)
    return readiness


def accept_phase6_stop_point_b(benchmark_root: str | Path) -> Path:
    """Validate the completed pilot and freeze the Phase-6 production plan."""
    root = Path(benchmark_root).resolve()
    if _phase_label_for_run(root) != "phase6":
        raise ValueError("This Stop Point B gate is specific to Phase 6.")
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "pilot":
        raise ValueError("Phase-6 Stop Point B applies only to a pilot run.")
    tasks = pd.read_csv(root / "benchmark_task_manifest.csv")
    enabled = tasks.loc[tasks["status"].ne("skipped_with_reason")].copy()
    metrics = pd.read_csv(root / "pilot_metrics.csv")
    truth = pd.read_csv(root / "pilot_truth_summary.csv")
    distribution = pd.read_csv(root / "pilot_distribution_comparison.csv")
    bayesian = pd.read_csv(root / "summaries" / "bayesian_diagnostics_summary.csv")
    blockers: list[str] = []
    if enabled["status"].isin(["pending", "running"]).any():
        blockers.append("pilot_tasks_not_terminal")
    if not enabled["status"].eq("success").all():
        blockers.append("pilot_enabled_task_failure")
    if not parse_boolean_series(distribution["within_tolerant_band"]).eq(True).all():
        blockers.append("pilot_distribution_outside_tolerant_band")
    scenarios = load_scenario_matrix(root / "benchmark_config" / "scenario_matrix.csv")
    effect_map = scenarios.set_index("scenario_id")["effect_strength"].astype(str).to_dict()
    for row in truth.itertuples(index=False):
        if effect_map[str(row.scenario_id)] == "null":
            if int(row.number_nonreference_true) != 0 or int(row.number_nonreference_null) == 0:
                blockers.append("null_truth_invalid")
                break
        elif int(row.number_nonreference_true) == 0 or int(row.number_nonreference_null) == 0:
            blockers.append("nonnull_truth_lacks_true_or_null")
            break
    formal_moderate = metrics.loc[
        metrics["scenario_id"].astype(str).eq("moderate_medium_calibrated")
        & metrics["benchmark_role"].astype(str).eq("formal_candidate")
    ]
    if pd.to_numeric(formal_moderate["mean_Power"], errors="coerce").fillna(0).le(0).all():
        blockers.append("all_formal_methods_zero_power_in_moderate")
    strong = metrics.loc[
        metrics["scenario_id"].astype(str).eq("strong_large_calibrated")
        & metrics["benchmark_role"].astype(str).eq("formal_candidate")
    ]
    if pd.to_numeric(strong["mean_FPR"], errors="coerce").gt(0.20).any():
        blockers.append("broad_strong_scenario_fpr")
    if (
        bayesian["status"].astype(str).ne("success").mean() > 0.5
        or pd.to_numeric(bayesian["divergences"], errors="coerce").fillna(1).gt(0).mean() > 0.5
    ):
        blockers.append("majority_sccoda_diagnostics_invalid")
    if not enabled.groupby(["scenario_id", "replicate_id"])["input_hash"].nunique().eq(1).all():
        blockers.append("pilot_input_hash_mismatch")
    figure_count = len(list((root / "pilot_figures").glob("*.png")))
    if figure_count < 8:
        blockers.append("pilot_figures_incomplete")

    runtime = pd.to_numeric(enabled["runtime_seconds"], errors="coerce")
    projected_serial_hours = float(runtime.sum() / 3 * 20 / 3600)
    plan_path = root / "production_plan.md"
    plan_lines = [
        "# Phase 6 production plan",
        "",
        "Stop Point B: self-accepted" if not blockers else "Stop Point B: blocked",
        "",
        "- Production benchmark ID: `phase6-production-v1`.",
        "- Fixed scale: 20 replicates per each of 7 preregistered scenarios.",
        "- Enabled methods: Propeller, DCATS, CLR_LMM, sccomp, scCODA, and the naive Welch sanity check.",
        "- Total enabled tasks: 840; every scenario × replicate uses one frozen input and truth table.",
        "- 10% checkpoint: r001-r002 after all six methods are terminal.",
        "- 50% checkpoint: r001-r010 after all six methods are terminal.",
        "- Final batch: r011-r020, followed by Stop Point C, canonical evaluation, and plotting.",
        "- scCODA runs one replicate ID (7 scenarios) per parent process; other methods may use larger method batches.",
        "- No decision threshold, truth rule, reference, scenario, or sampling parameter changes after this gate.",
        f"- Pilot-derived serial runtime projection: approximately {projected_serial_hours:.1f} hours.",
        "- Monitoring uses only task-manifest summaries at method-batch boundaries and the 10%/50% checkpoints.",
        "",
        "Pilot caution: sccomp showed elevated FPR in some non-null/heterogeneous scenarios. "
        "This remains a reported method result; it is not corrected by changing thresholds.",
        "",
        f"Blockers: {', '.join(blockers) if blockers else 'none'}.",
        "",
    ]
    plan_path.write_text("\n".join(plan_lines), encoding="utf-8")
    if blockers:
        raise RuntimeError(f"Phase-6 Stop Point B blockers: {blockers}")
    manifest["stop_point_b"] = {
        "status": "self_accepted",
        "created_at": _utc_now(),
        "production_replicates_per_scenario": 20,
        "production_plan": "production_plan.md",
    }
    manifest["status"] = "stop_point_b_self_accepted"
    _write_json(manifest_path, manifest)
    return plan_path


def record_interrupted_tasks(
    benchmark_root: str | Path,
    *,
    method: str,
    reason: str,
) -> Path:
    """Atomically close tasks left running by an external parent-process interruption."""
    root = Path(benchmark_root).resolve()
    task_path = root / "benchmark_task_manifest.csv"
    tasks = pd.read_csv(task_path)
    for column in ("failure_reason", "started_at", "finished_at", "result_subdir"):
        tasks[column] = tasks[column].astype("object")
    indices = tasks.index[
        tasks["method"].astype(str).eq(method) & tasks["status"].eq("running")
    ].tolist()
    if not indices:
        raise ValueError("No running task matches the requested method.")
    for index in indices:
        task = tasks.loc[index]
        method_root = _replicate_paths(
            root, str(task["scenario_id"]), str(task["replicate_id"])
        )["methods"] / method
        stderr = method_root / "stderr.txt"
        if not stderr.exists():
            stderr.write_text(reason + "\n", encoding="utf-8")
        stdout = method_root / "stdout.txt"
        if not stdout.exists():
            stdout.write_text("", encoding="utf-8")
        finished = pd.Timestamp.now(tz="UTC")
        started = pd.to_datetime(task["started_at"], utc=True, errors="coerce")
        runtime_seconds = (
            float((finished - started).total_seconds()) if pd.notna(started) else np.nan
        )
        task_manifest = {
            "benchmark_id": task["benchmark_id"],
            "scenario_id": task["scenario_id"],
            "replicate_id": task["replicate_id"],
            "run_id": task["run_id"],
            "analysis_id": task["analysis_id"],
            "method": method,
            "benchmark_role": task["benchmark_role"],
            "scientific_status": task["scientific_status"],
            "status": "runtime_failed",
            "failure_reason": reason,
            "input_hash": task["input_hash"],
            "runtime_seconds": runtime_seconds,
            "memory_measurement": "unavailable_after_external_parent_process_interruption",
            "attempt_count": int(task["attempt_count"]),
            "files": _file_records(method_root, excluded={"task_manifest.json"}),
        }
        _write_json(method_root / "task_manifest.json", task_manifest)
        tasks.at[index, "status"] = "runtime_failed"
        tasks.at[index, "failure_reason"] = reason
        tasks.at[index, "finished_at"] = finished.isoformat()
        tasks.at[index, "runtime_seconds"] = runtime_seconds
    _write_task_manifest(tasks, task_path)
    return task_path


def _mc_summary(replicates: pd.DataFrame, registry: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ("Power", "FPR", "Specificity", "Precision", "FDP_for_FDR")
    for (method, scenario_id), group in replicates.groupby(["method", "scenario_id"], sort=False):
        row: dict[str, Any] = {"method": method, "scenario_id": scenario_id}
        expected = int(group["replicate_id"].nunique())
        included = int(group["replicate_included"].sum())
        row.update({
            "number_expected_replicates": expected,
            "number_included_replicates": included,
            "number_missing_replicates": expected - included,
            "TP": int(group.loc[group["replicate_included"], "TP"].sum()),
            "FP": int(group.loc[group["replicate_included"], "FP"].sum()),
            "TN": int(group.loc[group["replicate_included"], "TN"].sum()),
            "FN": int(group.loc[group["replicate_included"], "FN"].sum()),
        })
        for metric in metrics:
            values = pd.to_numeric(group.loc[group["replicate_included"], metric], errors="coerce").dropna()
            mean = float(values.mean()) if len(values) else np.nan
            se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan
            row[f"mean_{metric}"] = mean
            row[f"mc_se_{metric}"] = se
            row[f"mc_ci_lower_{metric}"] = max(0.0, mean - 1.96 * se) if np.isfinite(se) else np.nan
            row[f"mc_ci_upper_{metric}"] = min(1.0, mean + 1.96 * se) if np.isfinite(se) else np.nan
            row[f"number_finite_{metric}"] = int(len(values))
        row["empirical_FDR"] = (
            row["mean_FDP_for_FDR"] if included == expected and expected > 0 else np.nan
        )
        row["empirical_FDR_reason"] = (
            None if included == expected and expected > 0 else "incomplete_replicates"
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    role_columns = registry[["method_id", "benchmark_role", "scientific_status"]].rename(
        columns={"method_id": "method"}
    )
    scenario_columns = scenarios[["scenario_id", "sample_size", "effect_strength", "difficulty"]]
    return result.merge(role_columns, on="method", how="left").merge(
        scenario_columns, on="scenario_id", how="left"
    )


def _stability_summary(
    tasks: pd.DataFrame,
    public: pd.DataFrame,
    truth: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    population = truth.loc[truth["truth_source"].eq("population")].copy()
    reference = population["cell_type"].astype(str).eq(population["reference_cell_type"].astype(str))
    expected_by_task = (
        population.loc[~reference].groupby(["scenario_id", "replicate_id"]).size().to_dict()
    )
    reference_by_run = (
        population[["run_id", "reference_cell_type"]]
        .drop_duplicates("run_id")
        .set_index("run_id")["reference_cell_type"]
        .astype(str)
        .to_dict()
    )
    rows = []
    enabled = tasks.loc[tasks["status"].ne("skipped_with_reason")]
    for (method, scenario_id), group in enabled.groupby(["method", "scenario_id"], sort=False):
        task_keys = set(zip(group["scenario_id"].astype(str), group["replicate_id"].astype(str)))
        expected = sum(expected_by_task.get(key, 0) for key in task_keys)
        subset = public.loc[
            public["method"].astype(str).eq(str(method))
            & public["run_id"].astype(str).isin(group["run_id"].astype(str))
            & public["effect_component"].eq("composition")
        ].copy()
        row_reference = subset["run_id"].astype(str).map(reference_by_run)
        nonreference = subset.loc[
            ~subset["cell_type"].astype(str).eq(row_reference.astype(str))
            & ~subset["contrast_status"].eq("reference")
        ]
        completed = len(nonreference)
        available = int(parse_boolean_series(nonreference.get("is_available", pd.Series(dtype=object))).eq(True).sum())
        valid = int(parse_boolean_series(nonreference.get("is_valid", pd.Series(dtype=object))).eq(True).sum())
        runtime_values = pd.to_numeric(group["runtime_seconds"], errors="coerce")
        number_tasks = int(len(group))
        rows.append({
            "method": method,
            "scenario_id": scenario_id,
            "expected_results": expected,
            "completed_results": completed,
            "available_results": available,
            "valid_results": valid,
            "missing_results": max(0, expected - completed),
            "runtime_failures": int(group["status"].eq("runtime_failed").sum()),
            "diagnostics_failures": int(group["status"].eq("diagnostics_invalid").sum()),
            "conversion_failures": int(group["status"].eq("conversion_failed").sum()),
            "completion_rate": completed / expected if expected else np.nan,
            "availability_rate": available / expected if expected else np.nan,
            "validity_rate": valid / expected if expected else np.nan,
            "diagnostics_invalid_rate": (
                float(group["status"].eq("diagnostics_invalid").sum()) / number_tasks
                if number_tasks else np.nan
            ),
            "runtime_failure_rate": (
                float(group["status"].eq("runtime_failed").sum()) / number_tasks
                if number_tasks else np.nan
            ),
            "conversion_failure_rate": (
                float(group["status"].eq("conversion_failed").sum()) / number_tasks
                if number_tasks else np.nan
            ),
            "median_runtime": float(runtime_values.median()),
            "runtime_q25": float(runtime_values.quantile(0.25)),
            "runtime_q75": float(runtime_values.quantile(0.75)),
            "runtime_q90": float(runtime_values.quantile(0.90)),
        })
    result = pd.DataFrame(rows)
    roles = registry[["method_id", "benchmark_role", "scientific_status"]].rename(
        columns={"method_id": "method"}
    )
    return result.merge(roles, on="method", how="left")


def _method_performance_summary(
    scenario_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    runtime: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact cross-scenario summary without changing evaluation units."""
    rows: list[dict[str, Any]] = []
    for method, group in scenario_metrics.groupby("method", sort=False):
        non_null = group.loc[group["effect_strength"].astype(str).ne("null")]
        method_stability = stability.loc[stability["method"].astype(str).eq(str(method))]
        method_runtime = runtime.loc[runtime["method"].astype(str).eq(str(method))]
        power = pd.to_numeric(non_null["mean_Power"], errors="coerce")
        fpr = pd.to_numeric(group["mean_FPR"], errors="coerce")
        fdr = pd.to_numeric(group["empirical_FDR"], errors="coerce")
        expected = float(method_stability["expected_results"].sum())
        rows.append({
            "method": method,
            "benchmark_role": group["benchmark_role"].iloc[0],
            "scientific_status": group["scientific_status"].iloc[0],
            "number_scenarios": int(group["scenario_id"].nunique()),
            "number_non_null_scenarios": int(non_null["scenario_id"].nunique()),
            "number_scenarios_with_complete_fdr": int(fdr.notna().sum()),
            "mean_power_across_non_null_scenarios": float(power.mean()) if power.notna().any() else np.nan,
            "mean_fpr_across_scenarios": float(fpr.mean()) if fpr.notna().any() else np.nan,
            "mean_empirical_fdr_across_complete_scenarios": (
                float(fdr.mean()) if fdr.notna().any() else np.nan
            ),
            "overall_completion_rate": (
                float(method_stability["completed_results"].sum()) / expected
                if expected > 0 else np.nan
            ),
            "overall_validity_rate": (
                float(method_stability["valid_results"].sum()) / expected
                if expected > 0 else np.nan
            ),
            "successful_tasks": int(method_runtime["status"].eq("success").sum()),
            "diagnostics_invalid_tasks": int(method_runtime["status"].eq("diagnostics_invalid").sum()),
            "runtime_failed_tasks": int(method_runtime["status"].eq("runtime_failed").sum()),
            "conversion_failed_tasks": int(method_runtime["status"].eq("conversion_failed").sum()),
            "median_runtime_seconds": float(
                pd.to_numeric(method_runtime["runtime_seconds"], errors="coerce").median()
            ),
            "total_runtime_seconds": float(
                pd.to_numeric(method_runtime["runtime_seconds"], errors="coerce").sum()
            ),
            "aggregation_note": (
                "Unweighted descriptive mean across registered scenarios; scenario-level rows and "
                "Monte Carlo uncertainty remain the inferential source."
            ),
        })
    return pd.DataFrame(rows)


def _plot_from_summaries(root: Path, *, pilot: bool) -> list[dict[str, str]]:
    import matplotlib.pyplot as plt

    summary = pd.read_csv(root / "summaries" / "scenario_level_metrics.csv")
    stability = pd.read_csv(root / "summaries" / "method_stability_summary.csv")
    runtime = pd.read_csv(root / "summaries" / "runtime_summary.csv")
    diagnostics = pd.read_csv(root / "summaries" / "diagnostic_failure_summary.csv")
    agreement = pd.read_csv(root / "summaries" / "method_agreement_summary.csv")
    bayesian = pd.read_csv(root / "summaries" / "bayesian_diagnostics_summary.csv")
    figure_root = root / ("pilot_figures" if pilot else "figures")
    figure_root.mkdir(exist_ok=False if pilot else True)
    figures: list[dict[str, str]] = []

    def save(fig: Any, stem: str, source: str) -> None:
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figure_root / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures.append({"figure": stem, "plot_ready_csv": source})

    methods = summary["method"].drop_duplicates().tolist()
    colors = dict(zip(methods, plt.cm.tab10(np.linspace(0, 1, max(1, len(methods))))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, title in zip(
        axes,
        ("mean_Power", "mean_FPR", "empirical_FDR"),
        ("Power", "FPR", "Empirical FDR"),
        strict=True,
    ):
        for method in methods:
            part = summary.loc[summary["method"].eq(method)]
            axis.plot(part["scenario_id"], part[metric], marker="o", label=method, color=colors[method])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=75)
        axis.set_ylim(-0.02, 1.02)
        if metric in {"mean_FPR", "empirical_FDR"}:
            axis.axhline(0.05, linestyle="--", color="black", linewidth=1)
    axes[0].legend(fontsize=7)
    save(fig, "method_performance_overview", "summaries/scenario_level_metrics.csv")

    heatmap = summary.pivot(index="method", columns="scenario_id", values="mean_Power")
    fig, ax = plt.subplots(
        figsize=(max(8, 1.25 * len(heatmap.columns)), max(3.5, 0.6 * len(heatmap.index)))
    )
    image = ax.imshow(
        np.ma.masked_invalid(heatmap.to_numpy(dtype=float)),
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    ax.set_xticks(np.arange(len(heatmap.columns)), heatmap.columns, rotation=70, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)), heatmap.index)
    ax.set_title("Scenario × method power")
    fig.colorbar(image, ax=ax, label="Power")
    save(fig, "scenario_method_power_heatmap", "summaries/scenario_level_metrics.csv")

    sanity_summary = summary.groupby(
        ["method", "benchmark_role"], sort=False, as_index=False
    ).agg(
        mean_power=("mean_Power", "mean"),
        mean_fpr=("mean_FPR", "mean"),
        mean_fdr=("empirical_FDR", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    role_colors = {
        "sanity_check": "tab:orange",
        "formal_candidate": "tab:blue",
        "legacy_comparator": "tab:gray",
    }
    bar_colors = [
        role_colors.get(str(value), "tab:gray")
        for value in sanity_summary["benchmark_role"]
    ]
    for axis, metric, title in zip(
        axes,
        ("mean_power", "mean_fpr", "mean_fdr"),
        ("Mean non-null power", "Mean FPR", "Mean empirical FDR"),
        strict=True,
    ):
        values = pd.to_numeric(sanity_summary[metric], errors="coerce")
        axis.bar(sanity_summary["method"], values, color=bar_colors)
        axis.tick_params(axis="x", rotation=68, labelsize=8)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
        if metric in {"mean_fpr", "mean_fdr"}:
            axis.axhline(0.05, linestyle="--", color="black", linewidth=1)
    save(fig, "sanity_check_comparison", "summaries/scenario_level_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    grouped = stability.groupby("method", sort=False).agg(
        completion_rate=("completion_rate", "mean"),
        validity_rate=("validity_rate", "mean"),
    )
    grouped.plot.bar(ax=axes[0], ylim=(0, 1.05), title="Completion and validity")
    runtime.groupby("method", sort=False)["runtime_seconds"].median().plot.bar(
        ax=axes[1], title="Median runtime (s)"
    )
    diagnostics.groupby("method", sort=False)["diagnostic_failure"].mean().plot.bar(
        ax=axes[2], ylim=(0, 1.05), title="Diagnostic failure fraction"
    )
    save(fig, "stability_runtime_diagnostics", "summaries/method_stability_summary.csv")

    null = summary.loc[summary["effect_strength"].eq("null")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not null.empty:
        positions = np.arange(len(null))
        ax.bar(positions - 0.18, null["mean_FPR"], width=0.36, label="FPR")
        ax.bar(positions + 0.18, null["empirical_FDR"], width=0.36, label="FDR")
        ax.set_xticks(positions, null["method"], rotation=45, ha="right")
    ax.axhline(0.05, linestyle="--", color="black", label="nominal 0.05")
    ax.set_ylim(0, 1.02)
    ax.set_title("All-null calibration")
    ax.legend()
    save(fig, "null_calibration", "summaries/scenario_level_metrics.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    scenario_order = summary["scenario_id"].drop_duplicates().tolist()
    for method in methods:
        part = summary.loc[summary["method"].eq(method)].set_index("scenario_id").reindex(scenario_order)
        ax.plot(scenario_order, part["mean_Power"], marker="o", label=method)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="x", rotation=70)
    ax.set_title("Power across registered scenarios")
    ax.legend(fontsize=7)
    save(fig, "scenario_power", "summaries/scenario_level_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not agreement.empty:
        agreement.groupby("scenario_id", sort=False)["all_methods_agree"].mean().plot.bar(ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Discovery agreement across methods")
    save(fig, "method_agreement", "summaries/method_agreement_summary.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    if not bayesian.empty:
        x = np.arange(len(bayesian))
        invalid = bayesian["status"].astype(str).ne("success")
        colors_bayes = np.where(invalid, "tab:red", "tab:blue")
        axes[0].scatter(
            x, pd.to_numeric(bayesian["divergences"], errors="coerce"),
            c=colors_bayes, alpha=0.75, s=18,
        )
        axes[1].scatter(
            x, pd.to_numeric(bayesian["r_hat_max"], errors="coerce"),
            c=colors_bayes, alpha=0.75, s=18,
        )
        axes[1].set_xticks(x, bayesian["scenario_id"], rotation=90, fontsize=6)
    axes[0].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_ylabel("Divergences")
    axes[0].set_title("scCODA sampling diagnostics (red = invalid diagnostic status)")
    axes[1].axhline(1.05, linestyle="--", color="black", linewidth=1)
    axes[1].set_ylabel("Maximum R-hat")
    save(
        fig,
        "bayesian_sampling_diagnostics",
        "summaries/plot_ready_bayesian_diagnostics.csv",
    )

    config = {
        "input_policy": "persisted_summary_tables_only",
        "formats": ["png", "pdf"],
        "nominal_reference": 0.05,
        "effect_magnitude_cross_method_plot": False,
    }
    (figure_root / "plot_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return figures


def regenerate_figures_from_summaries(benchmark_root: str | Path) -> Path:
    """Regenerate review figures from frozen plot-ready summary tables only."""
    root = Path(benchmark_root).resolve()
    if not (root / "summaries" / "scenario_level_metrics.csv").is_file():
        raise FileNotFoundError("Plot-ready benchmark summaries are missing.")
    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    pilot = str(manifest["phase"]) == "pilot"
    figures = _plot_from_summaries(root, pilot=pilot)
    final_report = root / "final_report"
    final_report.mkdir(exist_ok=True)
    figure_index = ["# Figure index", ""]
    for record in figures:
        figure_index.append(
            f"- `{record['figure']}.png` / `.pdf` — source `{record['plot_ready_csv']}`"
        )
    index_path = final_report / "figure_index.md"
    index_path.write_text("\n".join(figure_index) + "\n", encoding="utf-8")
    source_root = root / ("pilot_figures" if pilot else "figures")
    final_figures = final_report / "figures"
    final_figures.mkdir(exist_ok=True)
    for figure in source_root.iterdir():
        if figure.is_file():
            shutil.copyfile(figure, final_figures / figure.name)
    manifest["figures_regenerated_at"] = _utc_now()
    manifest["figure_source"] = "frozen_plot_ready_summaries"
    _write_json(root / "benchmark_manifest.json", manifest)
    return index_path


def _write_pilot_validation_tables(root: Path, tasks: pd.DataFrame) -> None:
    """Persist simulation/truth/status checks used for the Phase-6 pilot gate."""
    summary_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for (scenario_id, replicate_id), _ in tasks.groupby(
        ["scenario_id", "replicate_id"], sort=False
    ):
        paths = _replicate_paths(root, str(scenario_id), str(replicate_id))
        abundance = pd.read_csv(paths["simulation"] / "canonical_abundance_input.csv")
        samples = pd.read_csv(paths["simulation"] / "sample_metadata.csv")
        truth = pd.read_csv(paths["truth"] / "truth_table.csv")
        manifest = json.loads(
            (paths["manifests"] / "run_manifest.json").read_text(encoding="utf-8")
        )
        counts = abundance.pivot(index="sample_id", columns="cell_type", values="count")
        proportions = abundance.pivot(
            index="sample_id", columns="cell_type", values="proportion"
        )
        depths = counts.sum(axis=1)
        means = proportions.mean(axis=0)
        variances = proportions.var(axis=0, ddof=1)
        concentration = (
            depths.median() * (1 - variances / (means * (1 - means)))
            / ((variances / (means * (1 - means))) * depths.median() - 1)
        ).where(lambda values: values.gt(0) & np.isfinite(values))
        log_counts = np.log(counts.add(0.5))
        clr = log_counts.subtract(log_counts.mean(axis=1), axis=0)
        donor = samples.set_index("sample_id").loc[clr.index, "donor_id"].astype(str)
        donor_noise = clr.groupby(donor).mean().std(ddof=1).median()
        sample_heterogeneity = np.sqrt(
            clr.subtract(clr.median()).pow(2).mean(axis=1)
        ).median()
        summary_rows.append({
            "scenario_id": scenario_id,
            "replicate_id": replicate_id,
            "input_hash": manifest["input_hash"],
            "number_samples": int(len(samples)),
            "number_cell_types": int(counts.shape[1]),
            "total_count_mean": float(depths.mean()),
            "total_count_median": float(depths.median()),
            "total_count_sd": float(depths.std(ddof=1)),
            "zero_frequency": float(counts.eq(0).to_numpy().mean()),
            "low_abundance_q10": float(means.quantile(0.10)),
            "between_sample_variance_median": float(variances.median()),
            "estimated_dm_concentration_median": float(concentration.median()),
            "donor_clr_noise_sd": float(donor_noise),
            "sample_level_heterogeneity": float(sample_heterogeneity),
            "maximum_proportion_sum_error": float(
                (proportions.sum(axis=1) - 1).abs().max()
            ),
        })
        population = truth.loc[truth["truth_source"].astype(str).eq("population")].copy()
        flags = parse_boolean_series(population["is_true_effect"])
        reference = population["cell_type"].astype(str).eq(
            population["reference_cell_type"].astype(str)
        )
        truth_rows.append({
            "scenario_id": scenario_id,
            "replicate_id": replicate_id,
            "number_population_true": int(flags.eq(True).sum()),
            "number_population_null": int(flags.eq(False).sum()),
            "number_nonreference_true": int(flags.loc[~reference].eq(True).sum()),
            "number_nonreference_null": int(flags.loc[~reference].eq(False).sum()),
            "reference_cell_type": str(population["reference_cell_type"].iloc[0]),
            "reference_is_true": bool(flags.loc[reference].eq(True).any()),
        })
    simulation_summary = pd.DataFrame(summary_rows)
    truth_summary = pd.DataFrame(truth_rows)
    simulation_summary.to_csv(root / "pilot_simulation_summary.csv", index=False)
    truth_summary.to_csv(root / "pilot_truth_summary.csv", index=False)

    config = yaml.safe_load(_run_config_path(root).read_text(encoding="utf-8"))
    calibration_manifest = Path(str(config["calibration_manifest"]))
    if not calibration_manifest.is_absolute():
        calibration_manifest = REPOSITORY_ROOT / calibration_manifest
    calibration_root = calibration_manifest.parent
    layer_parameters = pd.read_csv(
        calibration_root
        / "parameter_estimation"
        / "layer1_tcell"
        / "estimated_parameters.csv"
    ).set_index("parameter")
    scenarios = pd.read_csv(root / "benchmark_config" / "scenario_matrix.csv")
    primary_scenario = "null_medium_calibrated"
    configured = scenarios.set_index("scenario_id").loc[primary_scenario]
    observed = simulation_summary.loc[
        simulation_summary["scenario_id"].eq(primary_scenario)
    ]
    metric_map = {
        "n_celltypes": "number_cell_types",
        "total_count_mean": "total_count_mean",
        "total_count_sd": "total_count_sd",
        "zero_frequency": "zero_frequency",
        "low_abundance_q10": "low_abundance_q10",
        "baseline_alpha_scale": "estimated_dm_concentration_median",
        "donor_noise_sd": "donor_clr_noise_sd",
        "sample_level_heterogeneity": "sample_level_heterogeneity",
    }
    comparison_rows: list[dict[str, Any]] = []
    for parameter, observed_column in metric_map.items():
        raw = float(layer_parameters.loc[parameter, "raw_estimate"])
        tolerant = float(layer_parameters.loc[parameter, "simulation_value"])
        scenario_value = (
            float(configured[parameter])
            if parameter in configured.index and pd.notna(configured[parameter])
            else tolerant
        )
        pilot_value = float(pd.to_numeric(observed[observed_column], errors="coerce").mean())
        use_observed_raw = parameter in {"donor_noise_sd", "sample_level_heterogeneity"}
        acceptance_reference = raw if use_observed_raw else scenario_value
        acceptance_basis = (
            "real_data_observed_moment"
            if use_observed_raw else "scenario_configured_value"
        )
        ratio = (
            pilot_value / acceptance_reference
            if acceptance_reference != 0 else np.nan
        )
        within = (
            bool(0.5 <= ratio <= 2.0)
            if np.isfinite(ratio)
            else bool(abs(pilot_value - scenario_value) <= 0.05)
        )
        if parameter in {"n_celltypes"}:
            within = bool(pilot_value == scenario_value)
        comparison_rows.append({
            "scenario_id": primary_scenario,
            "parameter": parameter,
            "real_data_raw_estimate": raw,
            "tolerant_simulation_value": tolerant,
            "scenario_configured_value": scenario_value,
            "pilot_observed_mean": pilot_value,
            "acceptance_reference": acceptance_reference,
            "acceptance_basis": acceptance_basis,
            "pilot_to_acceptance_reference_ratio": ratio,
            "tolerance_rule": (
                "exact" if parameter == "n_celltypes" else "0.5 <= observed/configured <= 2.0"
            ),
            "within_tolerant_band": within,
        })
    pd.DataFrame(comparison_rows).to_csv(
        root / "pilot_distribution_comparison.csv", index=False
    )
    status = (
        tasks.loc[tasks["status"].ne("skipped_with_reason")]
        .groupby(["method", "status"], sort=False)
        .size()
        .rename("number_tasks")
        .reset_index()
    )
    status.to_csv(root / "pilot_method_status.csv", index=False)


def evaluate_benchmark(benchmark_root: str | Path) -> Path:
    root = Path(benchmark_root).resolve()
    if any((root / "evaluation").iterdir()) or any((root / "summaries").iterdir()):
        raise FileExistsError("Evaluation or summary outputs already exist; benchmark outputs are immutable.")
    tasks = pd.read_csv(root / "benchmark_task_manifest.csv")
    pending = tasks["status"].isin(["pending", "running"])
    if pending.any():
        raise RuntimeError("All enabled method tasks must reach a terminal status before evaluation.")
    registry = load_method_registry(root / "method_registry" / "method_benchmark_registry.yaml")
    scenarios = load_scenario_matrix(root / "benchmark_config" / "scenario_matrix.csv")
    public_frames, evidence_frames, diagnostic_frames, truth_frames = [], [], [], []
    for (scenario_id, replicate_id), group in tasks.groupby(["scenario_id", "replicate_id"], sort=False):
        paths = _replicate_paths(root, str(scenario_id), str(replicate_id))
        truth_frames.append(pd.read_csv(paths["truth"] / "truth_table.csv"))
        enabled = group.loc[group["status"].ne("skipped_with_reason")]
        hashes = set(enabled["input_hash"].astype(str))
        if len(hashes) != 1:
            raise ValueError("Methods within a replicate do not share one canonical input hash.")
        for _, task in enabled.iterrows():
            method_root = paths["methods"] / str(task["method"])
            result_root = method_root / str(task.get("result_subdir", "."))
            if (result_root / "public_contrast.csv").is_file():
                public_frames.append(pd.read_csv(result_root / "public_contrast.csv"))
            if (result_root / "evidence.csv").is_file() and (result_root / "evidence.csv").stat().st_size > 1:
                try:
                    evidence_frames.append(pd.read_csv(result_root / "evidence.csv"))
                except pd.errors.EmptyDataError:
                    pass
            if (result_root / "diagnostics.csv").is_file():
                diagnostic_frames.append(pd.read_csv(result_root / "diagnostics.csv"))
    public = pd.concat(public_frames, ignore_index=True)
    evidence = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame()
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    truth = validate_truth_table(pd.concat(truth_frames, ignore_index=True))
    enabled_methods = tuple(registry.loc[registry["enabled"].astype(bool), "method_id"].astype(str))
    spec_doc = yaml.safe_load((root / "benchmark_config" / "evaluation_spec.yaml").read_text(encoding="utf-8"))
    policy_aliases = {
        "exclude_common_key_with_reason": "exclude_and_report",
    }
    spec = EvaluationSpec(
        truth_source=str(spec_doc["truth_source"]),
        required_effect_component=str(spec_doc["required_effect_component"]),
        eligible_estimand_levels=tuple(spec_doc["eligible_estimand_levels"]),
        method_universe_policy=str(spec_doc["method_universe_policy"]),
        missing_result_policy=policy_aliases.get(
            str(spec_doc["missing_result_policy"]), str(spec_doc["missing_result_policy"])
        ),
        invalid_result_policy=policy_aliases.get(
            str(spec_doc["invalid_result_policy"]), str(spec_doc["invalid_result_policy"])
        ),
        reference_policy=str(spec_doc["reference_policy"]),
        methods=enabled_methods,
    )
    evaluation = evaluate_contrasts(public, truth, spec)
    for name in ("aligned", "replicate_metrics", "aggregate_metrics", "stability"):
        (root / "evaluation" / name).mkdir()
    evaluation.aligned.to_csv(root / "evaluation" / "aligned" / "aligned_evaluation.csv", index=False)
    evaluation.replicate_metrics.to_csv(
        root / "evaluation" / "replicate_metrics" / "replicate_metrics.csv", index=False
    )
    scenario_metrics = _mc_summary(evaluation.replicate_metrics, registry, scenarios)
    scenario_metrics.to_csv(
        root / "evaluation" / "aggregate_metrics" / "scenario_level_metrics.csv", index=False
    )
    stability = _stability_summary(tasks, public, truth, registry)
    stability.to_csv(root / "evaluation" / "stability" / "method_stability.csv", index=False)

    roles = registry[["method_id", "benchmark_role", "scientific_status"]].rename(
        columns={"method_id": "method"}
    )
    runtime = tasks.loc[tasks["status"].ne("skipped_with_reason"), [
        "scenario_id", "replicate_id", "method", "status", "runtime_seconds",
        "failure_reason", "attempt_count", "result_subdir",
        "benchmark_role", "scientific_status",
    ]].copy()
    diag = runtime.copy()
    diag["diagnostic_failure"] = diag["status"].eq("diagnostics_invalid")
    diag["runtime_failure"] = diag["status"].eq("runtime_failed")
    diag["conversion_failure"] = diag["status"].eq("conversion_failed")
    population = truth.loc[truth["truth_source"].eq("population")]
    run_to_scenario = population[["run_id", "scenario_id", "replicate_id"]].drop_duplicates()
    bayesian_diagnostics = diagnostics.merge(run_to_scenario, on="run_id", how="left")
    parsed_details: list[dict[str, Any]] = []
    for value in bayesian_diagnostics.get("details", pd.Series(dtype=object)):
        if isinstance(value, dict):
            parsed_details.append(value)
            continue
        try:
            parsed = ast.literal_eval(str(value).replace("np.float64(", "("))
        except (ValueError, SyntaxError):
            parsed = {}
        parsed_details.append(parsed if isinstance(parsed, dict) else {})
    for field in (
        "r_hat_max", "ess_bulk_min", "ess_tail_min", "divergences",
        "acceptance_rate", "num_chains", "num_samples_per_chain",
    ):
        bayesian_diagnostics[field] = [details.get(field, np.nan) for details in parsed_details]
    bayesian_diagnostics = bayesian_diagnostics.loc[
        bayesian_diagnostics["method"].astype(str).eq("sccoda")
    ]
    decision = public.loc[public["effect_component"].eq("composition")].copy()
    decision = decision.merge(run_to_scenario, on="run_id", how="left", suffixes=("", "_truth"))
    decision["primary_decision"] = parse_boolean_series(decision["primary_decision"])
    decision = decision.loc[
        parse_boolean_series(decision["is_available"]).eq(True)
        & parse_boolean_series(decision["is_valid"]).eq(True)
    ]
    agreement_wide = decision.pivot_table(
        index=["scenario_id", "replicate_id", "cell_type"],
        columns="method",
        values="primary_decision",
        aggfunc="first",
    ).dropna()
    agreement = agreement_wide.reset_index()
    method_columns = [column for column in agreement_wide.columns]
    agreement["all_methods_agree"] = agreement_wide.nunique(axis=1).eq(1).to_numpy()
    agreement["positive_method_count"] = agreement_wide.astype(bool).sum(axis=1).to_numpy()

    summaries = root / "summaries"
    scenario_metrics.to_csv(summaries / "scenario_level_metrics.csv", index=False)
    stability.to_csv(summaries / "method_stability_summary.csv", index=False)
    runtime.to_csv(summaries / "runtime_summary.csv", index=False)
    diag.to_csv(summaries / "diagnostic_failure_summary.csv", index=False)
    agreement.to_csv(summaries / "method_agreement_summary.csv", index=False)
    scenario_metrics.loc[scenario_metrics["benchmark_role"].eq("sanity_check")].to_csv(
        summaries / "sanity_check_summary.csv", index=False
    )
    public.to_csv(summaries / "canonical_public_all.csv", index=False)
    evidence.to_csv(summaries / "native_evidence_all.csv", index=False)
    diagnostics.to_csv(summaries / "method_diagnostics_all.csv", index=False)
    bayesian_diagnostics.to_csv(summaries / "bayesian_diagnostics_summary.csv", index=False)
    method_performance = _method_performance_summary(
        scenario_metrics, stability, runtime
    )
    method_performance.to_csv(summaries / "method_performance_summary.csv", index=False)
    for source, target in (
        ("scenario_level_metrics.csv", "plot_ready_scenario_metrics.csv"),
        ("method_stability_summary.csv", "plot_ready_stability.csv"),
        ("runtime_summary.csv", "plot_ready_runtime.csv"),
        ("diagnostic_failure_summary.csv", "plot_ready_diagnostics.csv"),
        ("method_agreement_summary.csv", "plot_ready_agreement.csv"),
        ("bayesian_diagnostics_summary.csv", "plot_ready_bayesian_diagnostics.csv"),
    ):
        shutil.copyfile(summaries / source, summaries / target)

    pilot = str(json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))["phase"]) == "pilot"
    figures = _plot_from_summaries(root, pilot=pilot)
    if pilot:
        _write_pilot_validation_tables(root, tasks)
        shutil.copyfile(root / "benchmark_task_manifest.csv", root / "pilot_status.csv")
        shutil.copyfile(summaries / "scenario_level_metrics.csv", root / "pilot_metrics.csv")
        shutil.copyfile(summaries / "runtime_summary.csv", root / "pilot_runtime.csv")
        shutil.copyfile(summaries / "diagnostic_failure_summary.csv", root / "pilot_diagnostics.csv")
    figure_index = ["# Figure index", ""]
    for record in figures:
        figure_index.append(
            f"- `{record['figure']}.png` / `.pdf` — source `{record['plot_ready_csv']}`"
        )
    final_report = root / "final_report"
    (final_report / "figure_index.md").write_text(
        "\n".join(figure_index) + "\n", encoding="utf-8"
    )
    for name in (
        "method_performance_summary.csv",
        "method_stability_summary.csv",
        "scenario_level_metrics.csv",
        "sanity_check_summary.csv",
        "runtime_summary.csv",
        "diagnostic_failure_summary.csv",
    ):
        shutil.copyfile(summaries / name, final_report / name)
    final_figures = final_report / "figures"
    final_figures.mkdir()
    for figure in root.joinpath("figures").iterdir():
        if figure.is_file():
            shutil.copyfile(figure, final_figures / figure.name)
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "evaluation_and_figures_complete", "finished_at": _utc_now()})
    _write_json(manifest_path, manifest)
    phase_label = _phase_label_for_run(root)
    result_manifest = {
        "schema_version": f"{phase_label}-result-manifest-v1",
        "benchmark_id": tasks["benchmark_id"].iloc[0],
        "phase": "pilot" if pilot else "production",
        "status": "evaluation_and_figures_complete",
        "created_at": _utc_now(),
        "enabled_methods": list(enabled_methods),
        "tri_anchor_included": False,
        "common_input_hash_check": "passed",
        "files": _file_records(root, excluded={"final_report/result_manifest.json"}),
    }
    _write_json(root / "final_report" / "result_manifest.json", result_manifest)
    return root / "final_report" / "result_manifest.json"


def finalize_stop_point_d(benchmark_root: str | Path) -> Path:
    """Seal the human-review package after its narrative reports are written."""
    root = Path(benchmark_root).resolve()
    final_report = root / "final_report"
    phase_label = _phase_label_for_run(root)
    required = [
        f"{phase_label}_final_report.md",
        "method_performance_summary.csv",
        "method_stability_summary.csv",
        "scenario_level_metrics.csv",
        "sanity_check_summary.csv",
        "runtime_summary.csv",
        "diagnostic_failure_summary.csv",
        "figure_index.md",
        f"{phase_label}_remaining_questions.md",
    ]
    if phase_label == "phase6":
        required.extend([
            "stratified_parameter_report.md",
            "simulation_parameter_manifest.yaml",
            "method_completion_matrix.csv",
        ])
    missing = [name for name in required if not (final_report / name).is_file()]
    if not (final_report / "figures").is_dir():
        missing.append("figures/")
    if missing:
        raise FileNotFoundError(f"Stop Point D package is incomplete: {missing}")
    result_path = final_report / "result_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update({
        "status": "stop_point_d_awaiting_human_review",
        "created_at": _utc_now(),
        "files": _file_records(root, excluded={"final_report/result_manifest.json"}),
    })
    _write_json(result_path, result)
    benchmark_path = root / "benchmark_manifest.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark.update({
        "status": "stop_point_d_awaiting_human_review",
        "stop_point_d": {"status": "awaiting_human_review", "created_at": _utc_now()},
    })
    _write_json(benchmark_path, benchmark)
    result["files"] = _file_records(root, excluded={"final_report/result_manifest.json"})
    _write_json(result_path, result)
    return result_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Immutable differential-abundance benchmark orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("benchmark_id")
    initialize.add_argument("--phase", choices=("pilot", "production"), required=True)
    initialize.add_argument("--replicates", type=int, required=True)
    initialize.add_argument("--output-base", type=Path, default=REPOSITORY_ROOT / "benchmark_runs")
    initialize.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    initialize.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    initialize.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    initialize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    derive = subparsers.add_parser("derive-expanded")
    derive.add_argument("source_benchmark_root", type=Path)
    derive.add_argument("benchmark_id")
    derive.add_argument("--output-base", type=Path, default=REPOSITORY_ROOT / "benchmark_runs")
    derive.add_argument(
        "--registry", type=Path,
        default=REPOSITORY_ROOT / "config" / "phase6_method_benchmark_registry.yaml",
    )
    derive.add_argument(
        "--config", type=Path,
        default=REPOSITORY_ROOT / "config" / "phase6_benchmark.yaml",
    )
    run = subparsers.add_parser("run")
    run.add_argument("benchmark_root", type=Path)
    run.add_argument("--only-method", action="append", default=[])
    run.add_argument("--exclude-method", action="append", default=[])
    run.add_argument("--only-replicate", action="append", default=[])
    run_parallel = subparsers.add_parser("run-parallel")
    run_parallel.add_argument("benchmark_root", type=Path)
    run_parallel.add_argument("--workers", type=int, default=4)
    run_parallel.add_argument("--only-method", action="append", default=[])
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("benchmark_root", type=Path)
    retry = subparsers.add_parser("retry")
    retry.add_argument("benchmark_root", type=Path)
    retry.add_argument("--method", required=True)
    retry.add_argument("--reason-contains", required=True)
    migrate_dirichlet = subparsers.add_parser("migrate-dirichlet-estimand-contract")
    migrate_dirichlet.add_argument("benchmark_root", type=Path)
    regenerate_plots = subparsers.add_parser("regenerate-plots")
    regenerate_plots.add_argument("benchmark_root", type=Path)
    interrupted = subparsers.add_parser("record-interrupted")
    interrupted.add_argument("benchmark_root", type=Path)
    interrupted.add_argument("--method", required=True)
    interrupted.add_argument("--reason", required=True)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("benchmark_root", type=Path)
    checkpoint.add_argument("--replicate", action="append", required=True)
    stop_c = subparsers.add_parser("accept-stop-c")
    stop_c.add_argument("benchmark_root", type=Path)
    stop_b = subparsers.add_parser("accept-stop-b")
    stop_b.add_argument("benchmark_root", type=Path)
    finalize = subparsers.add_parser("finalize-stop-d")
    finalize.add_argument("benchmark_root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "initialize":
        result = initialize_benchmark(
            args.benchmark_id,
            phase=args.phase,
            replicates=args.replicates,
            output_base=args.output_base,
            registry_path=args.registry,
            scenario_path=args.scenarios,
            evaluation_path=args.evaluation,
            config_path=args.config,
        )
    elif args.command == "derive-expanded":
        result = derive_expanded_benchmark(
            args.source_benchmark_root, args.benchmark_id,
            output_base=args.output_base, registry_path=args.registry,
            config_path=args.config,
        )
    elif args.command == "run":
        result = run_pending_tasks(
            args.benchmark_root,
            only_methods=set(args.only_method) or None,
            exclude_methods=set(args.exclude_method) or None,
            only_replicates=set(args.only_replicate) or None,
        )
    elif args.command == "run-parallel":
        result = run_pending_tasks_parallel(
            args.benchmark_root, max_workers=args.workers,
            only_methods=set(args.only_method) or None,
        )
    elif args.command == "retry":
        result = retry_failed_tasks(
            args.benchmark_root,
            method=args.method,
            reason_contains=args.reason_contains,
        )
    elif args.command == "migrate-dirichlet-estimand-contract":
        result = migrate_dirichlet_estimand_contract(args.benchmark_root)
    elif args.command == "regenerate-plots":
        result = regenerate_figures_from_summaries(args.benchmark_root)
    elif args.command == "record-interrupted":
        result = record_interrupted_tasks(
            args.benchmark_root,
            method=args.method,
            reason=args.reason,
        )
    elif args.command == "checkpoint":
        result = production_checkpoint(
            args.benchmark_root,
            replicate_ids=set(args.replicate),
        )
    elif args.command == "accept-stop-c":
        result = accept_phase6_stop_point_c(args.benchmark_root)
    elif args.command == "accept-stop-b":
        result = accept_phase6_stop_point_b(args.benchmark_root)
    elif args.command == "finalize-stop-d":
        result = finalize_stop_point_d(args.benchmark_root)
    else:
        result = evaluate_benchmark(args.benchmark_root)
    print(result)


if __name__ == "__main__":
    main()
