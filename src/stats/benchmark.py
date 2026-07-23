from __future__ import annotations

import argparse
import ast
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
    if (frame["reference_cell_type"].astype(str) != "CT" + frame["n_celltypes"].astype(str)).any():
        raise ValueError("Phase-5 scenario references must be the protected final cell type.")
    return frame


def load_benchmark_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("generator") != "dirichlet_multinomial":
        raise ValueError("Phase 5 formal benchmark must use the DM generator.")
    if int(document.get("execution", {}).get("automatic_retries", -1)) != 0:
        raise ValueError("Phase 5 does not silently retry failed tasks.")
    return document


def _scenario_parameters(row: pd.Series) -> dict[str, Any]:
    return {
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
    phase_offset = 0 if phase == "pilot" else 100_000
    seed = (
        int(benchmark_config["base_seed"])
        + phase_offset
        + int(scenario.name) * 1000
        + replicate_number
    )
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
        "input_hash": canonical.input_hash(),
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
        "schema_version": "phase5-simulation-replicate-v1",
        "benchmark_id": benchmark_id,
        "run_id": run_id,
        "analysis_id": analysis_id,
        "scenario_id": scenario_id,
        "replicate_id": replicate_id,
        "simulation_seed": seed,
        "status": "frozen_before_method_execution",
        "input_hash": canonical.input_hash(),
        "created_at": _utc_now(),
        "files": _file_records(paths["root"], excluded={"manifests/run_manifest.json"}),
    }
    _write_json(paths["manifests"] / "run_manifest.json", manifest)
    return run_id, analysis_id, canonical.input_hash(), seed


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
        Path(config_path): root / "benchmark_config" / "phase5_benchmark.yaml",
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
        "schema_version": "phase5-benchmark-run-v1",
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
    config = yaml.safe_load((root / "benchmark_config" / "phase5_benchmark.yaml").read_text(encoding="utf-8"))
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
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    completed_fraction = len(replicate_ids) / int(manifest["replicates_per_scenario"])
    projected_bytes = int(total_bytes / completed_fraction) if completed_fraction else 0
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
    checkpoint_root = root / "diagnostics" / "production_checkpoint_20pct"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    status.to_csv(checkpoint_root / "task_status.csv", index=False)
    per_method.to_csv(checkpoint_root / "method_status.csv", index=False)
    truth_status.to_csv(checkpoint_root / "truth_distribution.csv", index=False)
    evaluation.replicate_metrics.to_csv(checkpoint_root / "replicate_metrics.csv", index=False)
    pd.DataFrame(hash_failures, columns=["scenario_id", "replicate_id", "reason"]).to_csv(
        checkpoint_root / "input_hash_failures.csv", index=False
    )
    report = {
        "checkpoint": "20_percent",
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
    if blockers:
        raise RuntimeError(f"Production checkpoint blockers: {blockers}")
    manifest["stop_point_c"] = {
        "status": "self_accepted",
        "checkpoint": "20_percent",
        "replicate_ids": sorted(replicate_ids),
        "created_at": report["created_at"],
    }
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
        (root / "benchmark_config" / "phase5_benchmark.yaml").read_text(encoding="utf-8")
    )
    candidates = tasks.index[
        tasks["method"].astype(str).eq(method)
        & tasks["status"].isin(["runtime_failed", "diagnostics_invalid"])
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
                    "corrected_failure_class": "runtime_failed",
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
            "validity_rate": valid / expected if expected else np.nan,
            "median_runtime": float(pd.to_numeric(group["runtime_seconds"], errors="coerce").median()),
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
    spec = EvaluationSpec(
        truth_source=str(spec_doc["truth_source"]),
        required_effect_component=str(spec_doc["required_effect_component"]),
        eligible_estimand_levels=tuple(spec_doc["eligible_estimand_levels"]),
        method_universe_policy=str(spec_doc["method_universe_policy"]),
        missing_result_policy=str(spec_doc["missing_result_policy"]),
        invalid_result_policy=str(spec_doc["invalid_result_policy"]),
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
    result_manifest = {
        "schema_version": "phase5-result-manifest-v1",
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
    required = (
        "phase5_final_report.md",
        "method_performance_summary.csv",
        "method_stability_summary.csv",
        "scenario_level_metrics.csv",
        "sanity_check_summary.csv",
        "runtime_summary.csv",
        "diagnostic_failure_summary.csv",
        "figure_index.md",
        "phase5_remaining_questions.md",
    )
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
    parser = argparse.ArgumentParser(description="Phase-5 immutable benchmark orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("benchmark_id")
    initialize.add_argument("--phase", choices=("pilot", "production"), required=True)
    initialize.add_argument("--replicates", type=int, required=True)
    initialize.add_argument("--output-base", type=Path, default=REPOSITORY_ROOT / "benchmark_runs")
    run = subparsers.add_parser("run")
    run.add_argument("benchmark_root", type=Path)
    run.add_argument("--only-method", action="append", default=[])
    run.add_argument("--exclude-method", action="append", default=[])
    run.add_argument("--only-replicate", action="append", default=[])
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("benchmark_root", type=Path)
    retry = subparsers.add_parser("retry")
    retry.add_argument("benchmark_root", type=Path)
    retry.add_argument("--method", required=True)
    retry.add_argument("--reason-contains", required=True)
    interrupted = subparsers.add_parser("record-interrupted")
    interrupted.add_argument("benchmark_root", type=Path)
    interrupted.add_argument("--method", required=True)
    interrupted.add_argument("--reason", required=True)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("benchmark_root", type=Path)
    checkpoint.add_argument("--replicate", action="append", required=True)
    finalize = subparsers.add_parser("finalize-stop-d")
    finalize.add_argument("benchmark_root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "initialize":
        result = initialize_benchmark(
            args.benchmark_id,
            phase=args.phase,
            replicates=args.replicates,
            output_base=args.output_base,
        )
    elif args.command == "run":
        result = run_pending_tasks(
            args.benchmark_root,
            only_methods=set(args.only_method) or None,
            exclude_methods=set(args.exclude_method) or None,
            only_replicates=set(args.only_replicate) or None,
        )
    elif args.command == "retry":
        result = retry_failed_tasks(
            args.benchmark_root,
            method=args.method,
            reason_contains=args.reason_contains,
        )
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
    elif args.command == "finalize-stop-d":
        result = finalize_stop_point_d(args.benchmark_root)
    else:
        result = evaluate_benchmark(args.benchmark_root)
    print(result)


if __name__ == "__main__":
    main()
