from __future__ import annotations

import re
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.stats.adapters._shared import (
    NativeAdapterError,
    NativeExecutor,
    NativeInput,
    bayesian_evidence,
    prepare_pairwise_input,
    public_row,
    require_columns,
)
from src.stats.adapters.base import BaseDifferentialAbundanceAdapter
from src.stats.schemas import CanonicalDAInput, DecisionRule, load_default_decision_rules


def _normalized_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    normalized = {_normalized_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        if _normalized_column_name(candidate) in normalized:
            return normalized[_normalized_column_name(candidate)]
    return None


def _load_sccoda_class():
    """Load Pertpy's scCODA implementation without importing unrelated tools.

    Pertpy 1.1.1 eagerly imports the full Scanpy/tool stack at package import
    time. On the audited Windows environment that cold import does not finish
    within five minutes, although scCODA itself only needs a small subset. This
    loader preserves the installed Pertpy implementation while avoiding those
    unrelated eager imports; Scanpy is stubbed because only `sample_level`
    AnnData input is used here.
    """
    if "pertpy.tools._coda._sccoda" in sys.modules:
        return sys.modules["pertpy.tools._coda._sccoda"].Sccoda

    spec = importlib.util.find_spec("pertpy")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("Pertpy is not installed.")
    pertpy_root = Path(next(iter(spec.submodule_search_locations)))
    package_paths = {
        "pertpy": pertpy_root,
        "pertpy.tools": pertpy_root / "tools",
        "pertpy.tools._coda": pertpy_root / "tools" / "_coda",
    }
    for package_name, package_path in package_paths.items():
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            package.__package__ = package_name
            if package_name == "pertpy":
                package.__version__ = "1.1.1"
            sys.modules[package_name] = package
    if "scanpy" not in sys.modules:
        scanpy_stub = types.ModuleType("scanpy")
        scanpy_stub.__doc__ = "Lazy stub for Pertpy sample-level scCODA execution."
        sys.modules["scanpy"] = scanpy_stub
    return importlib.import_module("pertpy.tools._coda._sccoda").Sccoda


class ScCODAAdapter(BaseDifferentialAbundanceAdapter):
    method_id = "sccoda"
    primary_decision_rule_id = "sccoda-native-credible-est-fdr-0.05-v1"

    def __init__(
        self,
        *,
        method_version: str = "unknown",
        executor: NativeExecutor | None = None,
        reference_cell_type: str | None = None,
        primary_pip_threshold: float = 0.95,
        native_est_fdr: float = 0.05,
        num_samples: int = 1000,
        num_warmup: int = 500,
        num_chains: int = 2,
        rng_key: int = 42,
        target_accept_prob: float = 0.98,
    ) -> None:
        super().__init__(method_version=method_version)
        if primary_pip_threshold != 0.95:
            raise ValueError("The v1 scCODA primary rule is registered at PIP threshold 0.95.")
        self.executor = executor
        self.reference_cell_type = reference_cell_type
        self.primary_pip_threshold = primary_pip_threshold
        self.native_est_fdr = native_est_fdr
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.rng_key = rng_key
        self.target_accept_prob = target_accept_prob

    def decision_rules(self) -> list[DecisionRule]:
        return load_default_decision_rules().for_method(self.method_id)

    def prepare_native_input(self, canonical_input: CanonicalDAInput, contrast: pd.Series) -> NativeInput:
        reference = contrast.get("reference_cell_type", self.reference_cell_type)
        if pd.isna(reference) or reference is None or str(reference) == "":
            reference = self.reference_cell_type
        included = set(canonical_input.cell_type_manifest.loc[
            canonical_input.cell_type_manifest["inclusion_status"].eq("included"), "cell_type"
        ].astype(str))
        if reference is None or str(reference) not in included:
            raise NativeAdapterError(
                "reference_cell_type_required",
                "scCODA requires an explicit included reference_cell_type for auditable effect semantics.",
            )
        return prepare_pairwise_input(canonical_input, contrast, options={
            "reference_cell_type": str(reference),
            "primary_pip_threshold": self.primary_pip_threshold,
            "native_est_fdr": self.native_est_fdr,
            "num_samples": self.num_samples,
            "num_warmup": self.num_warmup,
            "num_chains": self.num_chains,
            "rng_key": self.rng_key,
            "target_accept_prob": self.target_accept_prob,
        })

    def execute_native(self, native_input: NativeInput, contrast: pd.Series) -> pd.DataFrame:
        if self.executor is not None:
            return self.executor(native_input, contrast)
        return self._execute_pertpy(native_input)

    def _execute_pertpy(self, native_input: NativeInput) -> pd.DataFrame:
        try:
            import anndata as ad
            Sccoda = _load_sccoda_class()
        except ImportError as exc:
            raise NativeAdapterError(
                "dependency_unavailable",
                "Pertpy with the scCODA dependencies is not installed in this Python environment.",
            ) from exc

        factor = str(native_input.contrast["factor"])
        group_1 = str(native_input.contrast["group_1"])
        group_2 = str(native_input.contrast["group_2"])
        sample_order = native_input.sample_manifest["sample_id"].astype(str).tolist()
        cell_order = native_input.cell_type_manifest["cell_type"].astype(str).tolist()
        counts = native_input.abundance.pivot(
            index="sample_id", columns="cell_type", values="count"
        ).reindex(index=sample_order, columns=cell_order)
        if counts.isna().any().any():
            raise NativeAdapterError("invalid_native_input", "scCODA count matrix is incomplete.")
        obs = native_input.sample_manifest.set_index("sample_id").reindex(sample_order).copy()
        obs.index = pd.Index(obs.index.astype(str), name=None)
        obs[factor] = pd.Categorical(obs[factor].astype(str), categories=[group_2, group_1], ordered=True)
        sample_adata = ad.AnnData(
            X=counts.to_numpy(dtype=int),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(cell_order, name="cell_type")),
        )

        import mudata
        mudata.set_options(pull_on_update=False)
        model = Sccoda()
        mdata = model.load(sample_adata, type="sample_level")
        mdata["coda"].obs.index.name = None
        mdata["coda"].var.index.name = None
        formula = f"C({factor}, Treatment(reference={group_2!r}))"
        reference_cell_type = str(native_input.options["reference_cell_type"])
        model.prepare(mdata, formula=formula, reference_cell_type=reference_cell_type)
        mdata["coda"].obs.index.name = None
        mdata["coda"].var.index.name = None
        import jax.numpy as jnp
        from jax import random
        from numpyro.infer import MCMC, NUTS, initialization

        sample_adata = mdata["coda"]
        rng_seed = int(native_input.options["rng_key"])
        rng_key_array = random.key_data(random.key(rng_seed))
        sample_adata.uns["scCODA_params"]["mcmc"]["rng_key"] = np.array(rng_key_array)
        sample_adata = model.set_init_mcmc_states(
            rng_seed,
            sample_adata.uns["scCODA_params"]["reference_index"],
            sample_adata,
        )
        init_params = sample_adata.uns["scCODA_params"]["mcmc"]["init_params"]
        nuts_kernel = NUTS(
            model.model,
            init_strategy=initialization.init_to_value(values=init_params),
            target_accept_prob=float(native_input.options["target_accept_prob"]),
            max_tree_depth=15,
        )
        num_samples = int(native_input.options["num_samples"])
        num_warmup = int(native_input.options["num_warmup"])
        num_chains = int(native_input.options["num_chains"])
        sample_adata.uns["scCODA_params"]["mcmc"].update({
            "num_samples": num_samples,
            "num_warmup": num_warmup,
            "algorithm": "NUTS",
        })
        extra_fields = (
            "potential_energy", "num_steps", "adapt_state.step_size",
            "accept_prob", "mean_accept_prob", "diverging",
        )
        model.mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            chain_method="sequential",
        )
        model.mcmc.run(
            rng_key_array,
            jnp.array(sample_adata.X, dtype="float64"),
            jnp.array(sample_adata.obsm["covariate_matrix"], dtype="float64"),
            jnp.array(sample_adata.obsm["sample_counts"], dtype="float64"),
            jnp.array(sample_adata.uns["scCODA_params"]["reference_index"]),
            sample_adata,
            extra_fields=extra_fields,
        )
        samples = {key: np.asarray(value) for key, value in model.mcmc.get_samples().items()}
        sample_adata.uns["scCODA_params"]["mcmc"].update({
            "acceptance_rate": float(np.asarray(model.mcmc.last_state.mean_accept_prob).mean()),
            "samples": samples,
            "num_chains": num_chains,
            "extra_fields": {
                key.replace(".", "_"): np.asarray(value)
                for key, value in model.mcmc.get_extra_fields().items()
            },
        })
        intercept_df, effect_df = model.summary_prepare(sample_adata)
        sample_adata.varm["intercept_df"] = intercept_df
        for covariate in effect_df.index.get_level_values("Covariate"):
            sample_adata.varm[f"effect_df_{covariate}"] = effect_df.loc[covariate, :]
        model.set_fdr(mdata, est_fdr=float(native_input.options["native_est_fdr"]))
        effects_indexed = model.get_effect_df(mdata)
        credible = pd.Series(model.credible_effects(mdata)).reset_index(drop=True)
        effects = effects_indexed.reset_index()
        if len(credible) == len(effects):
            # Attach the native decision before contrast filtering so that it
            # remains aligned with the corresponding cell-type/effect row.
            effects["native_decision"] = credible.to_numpy()

        cell_column = _find_column(effects, "cell type", "cell_type")
        covariate_column = _find_column(effects, "covariate")
        effect_column = _find_column(effects, "final parameter")
        pip_column = _find_column(effects, "inclusion probability")
        if not all((cell_column, effect_column, pip_column)):
            raise NativeAdapterError(
                "invalid_native_result",
                f"Unrecognized Pertpy scCODA effect columns: {effects.columns.tolist()}",
            )
        if covariate_column is not None:
            matching = effects[covariate_column].astype(str).str.contains(re.escape(group_1), regex=True)
            if matching.any():
                effects = effects.loc[matching].copy()
        if effects.empty:
            raise NativeAdapterError("native_output_missing", "No scCODA row matched the requested contrast.")

        if "native_decision" not in effects.columns:
            effects["native_decision"] = pd.to_numeric(effects[effect_column], errors="coerce").ne(0)

        adata_coda = mdata["coda"]
        params = adata_coda.uns.get("scCODA_params", {})
        native_threshold = params.get("threshold_prob", np.nan)
        reference_index = params.get("reference_index")
        actual_reference = reference_cell_type
        if isinstance(reference_index, (int, np.integer)) and 0 <= int(reference_index) < len(cell_order):
            actual_reference = cell_order[int(reference_index)]

        lower_columns = [column for column in effects.columns if "hdi" in str(column).lower()]
        lower = lower_columns[0] if lower_columns else None
        upper = lower_columns[-1] if lower_columns else None
        result = pd.DataFrame({
            "cell_type": effects[cell_column].astype(str),
            "covariate": effects[covariate_column].astype(str) if covariate_column else formula,
            "final_parameter": pd.to_numeric(effects[effect_column], errors="coerce"),
            "posterior_inclusion_probability": pd.to_numeric(effects[pip_column], errors="coerce"),
            "native_decision": effects["native_decision"],
            "native_inclusion_threshold": native_threshold,
            "credible_interval_lower": pd.to_numeric(effects[lower], errors="coerce") if lower else np.nan,
            "credible_interval_upper": pd.to_numeric(effects[upper], errors="coerce") if upper else np.nan,
            "reference_cell_type": actual_reference,
            "native_est_fdr": float(native_input.options["native_est_fdr"]),
        })
        import arviz as az

        arviz_data = model.make_arviz(sample_adata, num_prior_samples=0, use_posterior_predictive=False)
        diagnostic_summary = az.summary(
            arviz_data,
            var_names=["beta"],
            kind="diagnostics",
        )
        rhat = pd.to_numeric(diagnostic_summary.get("r_hat"), errors="coerce")
        ess_bulk = pd.to_numeric(diagnostic_summary.get("ess_bulk"), errors="coerce")
        ess_tail = pd.to_numeric(diagnostic_summary.get("ess_tail"), errors="coerce")
        extra_fields = model.mcmc.get_extra_fields()
        divergences = int(np.asarray(extra_fields.get("diverging", [])).sum())
        acceptance_rate = float(np.asarray(model.mcmc.last_state.mean_accept_prob).mean())
        rhat_max = float(rhat.max()) if rhat.notna().any() else np.nan
        ess_bulk_min = float(ess_bulk.min()) if ess_bulk.notna().any() else np.nan
        ess_tail_min = float(ess_tail.min()) if ess_tail.notna().any() else np.nan
        converged = bool(
            np.isfinite(rhat_max)
            and rhat_max <= 1.05
            and np.isfinite(ess_bulk_min)
            and ess_bulk_min >= 50
            and np.isfinite(ess_tail_min)
            and ess_tail_min >= 50
            and divergences == 0
            and 0.6 <= acceptance_rate <= 1.0
        )
        result.attrs["diagnostics"] = {
            "reference_cell_type": actual_reference,
            "native_inclusion_threshold": native_threshold,
            "formula": formula,
            "converged": converged,
            "r_hat_max": rhat_max,
            "ess_bulk_min": ess_bulk_min,
            "ess_tail_min": ess_tail_min,
            "divergences": divergences,
            "acceptance_rate": acceptance_rate,
            "num_chains": num_chains,
            "num_samples_per_chain": num_samples,
            "num_warmup_per_chain": num_warmup,
            "target_accept_prob": float(native_input.options["target_accept_prob"]),
            "diagnostic_rule": (
                "r_hat<=1.05; ess_bulk>=50; ess_tail>=50; divergences=0; "
                "0.6<=acceptance_rate<=1.0"
            ),
        }
        return result

    def transform_native_output(
        self,
        native_output: pd.DataFrame,
        canonical_input: CanonicalDAInput,
        contrast: pd.Series,
        *,
        analysis_id: str,
        diagnostic_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        require_columns(
            native_output,
            {
                "cell_type", "final_parameter", "posterior_inclusion_probability",
                "native_decision", "reference_cell_type",
            },
            self.method_id,
        )
        public_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        for _, native in native_output.iterrows():
            cell_type = str(native["cell_type"])
            reference_cell_type = str(native["reference_cell_type"])
            reference_strategy = contrast.get("reference_strategy", "common_exclusion")
            if pd.isna(reference_strategy) or not str(reference_strategy):
                reference_strategy = "common_exclusion"
            selection_reason = contrast.get(
                "reference_selection_reason",
                "preregistered_fixed_reference_before_model_results",
            )
            if pd.isna(selection_reason) or not str(selection_reason):
                selection_reason = "preregistered_fixed_reference_before_model_results"
            row, result_id, evidence_id = public_row(
                method_id=self.method_id,
                method_version=self.method_version,
                analysis_id=analysis_id,
                diagnostic_id=diagnostic_id,
                contrast=contrast,
                cell_type=cell_type,
                effect_component="composition",
                estimate=native["final_parameter"],
                effect_estimand="relative_compositional_log_effect",
                effect_scale="log_ratio",
                direction_basis=f"log_ratio:null=0;reference_cell_type={reference_cell_type}",
                decision_rule_id=self.primary_decision_rule_id,
                reference_cell_type=reference_cell_type,
                reference_strategy=str(reference_strategy),
                reference_selection_reason=str(selection_reason),
                reference_is_fixed=True,
                result_interpretation=(
                    f"Composition effect for {contrast['group_1']} versus {contrast['group_2']} "
                    f"relative to reference cell type {reference_cell_type}; not an absolute abundance difference."
                ),
            )
            if cell_type == reference_cell_type:
                row.update({
                    "evidence_id": pd.NA,
                    "estimate": float("nan"),
                    "effect_direction": "not_applicable",
                    "primary_decision": pd.NA,
                    "decision_rule_id": pd.NA,
                    "is_available": False,
                    "is_valid": False,
                    "is_benchmark_eligible": False,
                    "contrast_status": "reference",
                    "failure_reason": pd.NA,
                })
                public_rows.append(row)
                continue

            public_rows.append(row)
            pip = native["posterior_inclusion_probability"]
            evidence_rows.append(bayesian_evidence(
                evidence_id=evidence_id,
                result_id=result_id,
                native_decision=native["native_decision"],
                native_metric="posterior_inclusion_probability",
                native_value=pip,
                native_rule_id=f"sccoda-native-est-fdr-{self.native_est_fdr}-v1",
                posterior_probability=pip,
                posterior_probability_type="inclusion",
                posterior_inclusion_probability=pip,
                credible_interval_lower=native.get("credible_interval_lower", pd.NA),
                credible_interval_upper=native.get("credible_interval_upper", pd.NA),
                extra={
                    "native_inclusion_threshold": native.get("native_inclusion_threshold", pd.NA),
                    "native_est_fdr": native.get("native_est_fdr", self.native_est_fdr),
                    "reference_cell_type": reference_cell_type,
                    "posterior_effect": native["final_parameter"],
                    "covariate": native.get("covariate", pd.NA),
                },
            ))
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)
