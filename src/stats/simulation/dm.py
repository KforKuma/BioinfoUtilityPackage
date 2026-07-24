from __future__ import annotations
import warnings
import logging
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.stats import (
    dirichlet,
    multinomial
)


from src.utils.hier_logger import logged
from src.stats.simulation.design import build_simulation_metadata, validate_factor_levels
from src.stats.simulation.truth_refine import attach_population_effects, refine_ground_truth_by_observation
logger = logging.getLogger(__name__)


# -----------------------
# 生成模拟数据：Dirichlet-Multinomial 模拟
# 有利于 Dirichlet 回归
# -----------------------
@logged
def simulate_DM_data(
        *,
        n_donors: int = 8,
        n_samples_per_donor: int = 4,
        n_celltypes: int = 50,
        cell_type_names: Sequence[str] | None = None,
        baseline_composition: Mapping[str, float] | Sequence[float] | None = None,
        baseline_alpha_scale: float = 30.0,
        disease_effect_size: float = 0.5,
        tissue_effect_size: float = 0.6,
        interaction_effect_size: float = 1.0,
        inflamed_cell_frac: float = 0.15,
        sampling_bias_strength: float = 0.0,
        disease_levels: Tuple[str, str, str] = ("HC", "CD", "UC"),
        tissue_levels: Tuple[str, str] = ("nif", "if"),
        total_count_mean=2e4,total_count_sd=5e2,min_count=1000,
        donor_noise_sd: float = 0.3,
        assignment_strategy: str = "balanced",
        protected_cell_types: Sequence[str] = (),
        population_null_tolerance: float = 1e-12,
        population_reference_cell_type: str | None = None,
        random_state: int = 1234
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成 Dirichlet-Multinomial 组成丰度模拟数据。

    该模拟器以 Dirichlet alpha 作为底层组成参数，再从 multinomial 中采样 count。
    它适合生成对 Dirichlet/Dirichlet-Multinomial engine 友好的模拟数据，也会返回
    一张 refined ground truth 表，用于评估各方法对真实注入效应的恢复能力。

    Args:
        n_donors: donor 数量。
        n_samples_per_donor: 每个 donor 的 sample 数量。
        n_celltypes: cell subtype/subpopulation 数量。
        baseline_alpha_scale: baseline Dirichlet alpha 总量尺度。
        disease_effect_size: disease 主效应的 log 尺度大小。
        tissue_effect_size: tissue 主效应的 log 尺度大小。
        interaction_effect_size: disease x tissue 交互效应大小。
        inflamed_cell_frac: 受 tissue/interaction 影响的亚群比例。
        sampling_bias_strength: 样本级采样偏差强度。
        disease_levels: disease 水平，首个元素作为参考组。
        tissue_levels: tissue 水平，首个元素作为参考组。
        total_count_mean: 每个 sample 的测序深度均值。
        total_count_sd: 每个 sample 的测序深度标准差。
        min_count: 每个 sample 的最小测序深度。
        donor_noise_sd: donor 级别 alpha 扰动强度。
        assignment_strategy: ``balanced`` 为默认的分层随机平衡设计；``random``
            保留历史独立随机分配行为。
        protected_cell_types: 不得承载 latent injection 的预注册 cell type，例如
            reference cell type。
        population_null_tolerance: 判定精确 population effect 数值零的显式容差。
        population_reference_cell_type: population log-ratio estimand 的预注册 reference。
        random_state: 随机种子。

    Returns:
        ``(df_long, df_true_refined)``。``df_long`` 是模拟出的长表 count 数据；
        ``df_true_refined`` 是按实际观测 LFC 修正后的 ground truth。

    Example:
        >>> df_sim, df_truth = simulate_DM_data(
        ...     n_donors=12,
        ...     n_samples_per_donor=3,
        ...     n_celltypes=40,
        ...     disease_effect_size=0.5,
        ...     tissue_effect_size=0.6,
        ...     disease_levels=("HC", "CD", "UC"),
        ...     tissue_levels=("nif", "if"),
        ...     random_state=2024,
        ... )
        >>> df_sim.head()
        # donor_id/sample_id/disease/tissue/cell_type/count/total_count/prop
        >>> df_truth.query("True_Significant").head()
        # 用于评估模拟数据中被注入且可观察到的真实差异。
    """
    if not isinstance(n_celltypes, int) or isinstance(n_celltypes, bool) or n_celltypes <= 1:
        raise ValueError("`n_celltypes` must be an integer greater than 1.")
    disease_levels, tissue_levels = validate_factor_levels(disease_levels, tissue_levels)
    numeric_parameters = np.asarray([
        baseline_alpha_scale, disease_effect_size, tissue_effect_size,
        interaction_effect_size, inflamed_cell_frac, sampling_bias_strength,
        total_count_mean, total_count_sd, min_count, donor_noise_sd,
        population_null_tolerance,
    ], dtype=float)
    if not np.isfinite(numeric_parameters).all():
        raise ValueError("Simulation numeric parameters must be finite.")
    if not 0 <= inflamed_cell_frac <= 1:
        raise ValueError("`inflamed_cell_frac` must be between 0 and 1.")
    if baseline_alpha_scale <= 0:
        raise ValueError("`baseline_alpha_scale` must be greater than 0.")
    if min(disease_effect_size, tissue_effect_size, interaction_effect_size) < 0:
        raise ValueError("Effect-size parameters are non-negative magnitudes.")
    if min(sampling_bias_strength, donor_noise_sd, total_count_sd) < 0:
        raise ValueError("Noise and sampling-bias scale parameters must be non-negative.")
    if total_count_mean <= 0 or min_count <= 0:
        raise ValueError("`total_count_mean` and `min_count` must be greater than 0.")

    rng = np.random.default_rng(random_state)
    if cell_type_names is None:
        cell_type_names = [f"CT{i + 1}" for i in range(n_celltypes)]
    else:
        cell_type_names = [str(value) for value in cell_type_names]
        if len(cell_type_names) != n_celltypes or len(set(cell_type_names)) != n_celltypes:
            raise ValueError("`cell_type_names` must contain n_celltypes unique values.")
    if (
        population_reference_cell_type is not None
        and str(population_reference_cell_type) not in set(map(str, protected_cell_types))
    ):
        raise ValueError("The population reference cell type must also be protected from injection.")
    
    # 1. Baseline alpha
    if baseline_composition is None:
        baseline = rng.uniform(0.5, 2.0, n_celltypes)
    elif isinstance(baseline_composition, Mapping):
        missing_baseline = set(cell_type_names) - set(map(str, baseline_composition))
        if missing_baseline:
            raise ValueError(
                "`baseline_composition` mapping is missing cell types: "
                f"{sorted(missing_baseline)}"
            )
        baseline = np.asarray(
            [baseline_composition[str(cell_type)] for cell_type in cell_type_names], dtype=float
        )
    else:
        baseline = np.asarray(list(baseline_composition), dtype=float)
        if len(baseline) != n_celltypes:
            raise ValueError("`baseline_composition` must have n_celltypes values.")
    if not np.isfinite(baseline).all() or (baseline <= 0).any():
        raise ValueError("`baseline_composition` values must be finite and positive.")
    baseline = baseline / baseline.sum() * baseline_alpha_scale
    
    # 1.5 效应向量 (假设该函数已定义)
    disease_main_effects_dict, tissue_effect_vec, interaction_effects_dict, df_true_effect = build_DM_effects_with_main_effect(
        cell_type_names, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng, protected_cell_types=protected_cell_types
    )
    
    # 2. 预准备元数据和采样 (为了提速，先生成所有参数)
    ref_disease = disease_levels[0]
    ref_tissue = tissue_levels[0]
    
    # 提前生成采样偏差向量
    latent_axis = np.zeros(n_celltypes)
    if sampling_bias_strength > 0:
        latent_axis = rng.normal(0, 1, n_celltypes)
        latent_axis /= np.linalg.norm(latent_axis)
    
    df_meta = build_simulation_metadata(
        n_donors=n_donors,
        n_samples_per_donor=n_samples_per_donor,
        disease_levels=disease_levels,
        tissue_levels=tissue_levels,
        rng=rng,
        assignment_strategy=assignment_strategy,
    )
    donor_alpha = {
        donor: baseline * np.exp(rng.normal(0, donor_noise_sd, n_celltypes))
        for donor in df_meta["donor_id"].unique()
    }
    counts_list = []
    for sample in df_meta.itertuples(index=False):
        alpha = donor_alpha[sample.donor_id].copy()

        if sample.disease != ref_disease:
            alpha *= np.exp(disease_main_effects_dict[sample.disease])
        if sample.tissue != ref_tissue:
            alpha *= np.exp(tissue_effect_vec)
            if sample.disease != ref_disease:
                alpha *= np.exp(interaction_effects_dict[sample.disease])

        if sampling_bias_strength > 0:
            alpha *= np.exp(rng.normal(0, sampling_bias_strength) * latent_axis)

        alpha = np.maximum(alpha, 1e-6)
        N = int(max(round(rng.normal(total_count_mean, total_count_sd)), min_count))
        p = rng.dirichlet(alpha)
        counts_list.append(rng.multinomial(n=N, pvals=p))
    
    # 3. 内存友好型构建长表 (核心优化点)
    counts_matrix = np.vstack(counts_list)
    
    # 直接利用 NumPy 向量化展开，避开 melt
    df_long = df_meta.iloc[np.repeat(np.arange(len(df_meta)), n_celltypes)].copy()
    df_long['cell_type'] = np.tile(cell_type_names, len(df_meta))
    df_long['count'] = counts_matrix.flatten()
    
    # 4. 计算比例
    df_long['total_count'] = df_long.groupby('sample_id')['count'].transform('sum')
    df_long['prop'] = df_long['count'] / df_long['total_count']
    
    df_true_effect = attach_population_effects(
        df_true_effect,
        baseline_logits=np.log(baseline),
        cell_types=cell_type_names,
        disease_effects=disease_main_effects_dict,
        tissue_effect=tissue_effect_vec,
        interaction_effects=interaction_effects_dict,
        population_null_tolerance=population_null_tolerance,
        population_reference_cell_type=population_reference_cell_type,
    )
    df_long = df_long.reset_index(drop=True)
    df_true_refined = refine_ground_truth_by_observation(
        df_long, df_true_effect, injected_effect_scale="log_alpha_effect"
    )
    return df_long, df_true_refined


def build_DM_effects_with_main_effect(
        cell_type_names, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng, protected_cell_types: Sequence[str] = ()
):
    """构建 DM 模拟所需的主效应、组织效应和交互效应。

    该函数会为 disease main effect、tissue main effect 和 disease x tissue
    interaction 分别构建 log 尺度效应向量。效应方向允许增加或减少；交互项的
    ground truth 使用全局 ``ref_disease x ref_tissue`` 作为参照组，保留原始语义。

    Args:
        cell_type_names: cell subtype/subpopulation 名称列表。
        disease_levels: disease 水平，首个元素为参考组。
        tissue_levels: tissue 水平，首个元素为参考组。
        disease_effect_size: disease 主效应大小。
        tissue_effect_size: tissue 主效应大小。
        interaction_effect_size: 交互效应大小。
        inflamed_cell_frac: 受 tissue/interaction 影响的亚群比例。
        rng: ``numpy.random.Generator`` 实例。

    Returns:
        ``(disease_main_effects_dict, tissue_effect_vec, interaction_effects_dict,
        df_true_effect)``。

    Example:
        >>> rng = np.random.default_rng(1)
        >>> effects, tissue_vec, inter, truth = build_DM_effects_with_main_effect(
        ...     ["CT1", "CT2", "CT3"],
        ...     ("HC", "CD"),
        ...     ("nif", "if"),
        ...     0.4,
        ...     0.6,
        ...     0.8,
        ...     0.3,
        ...     rng,
        ... )
        >>> truth[["cell_type", "contrast_factor", "True_Effect"]].head()
        # 记录每个亚群和对比的真实注入效应。
    """
    if len(tissue_levels) < 2:
        raise ValueError("`tissue_levels` must contain at least two levels.")
    n_celltypes = len(cell_type_names)
    if n_celltypes == 0:
        raise ValueError("`cell_type_names` must not be empty.")
    protected = set(map(str, protected_cell_types))
    unknown_protected = protected - set(map(str, cell_type_names))
    if unknown_protected:
        raise ValueError(f"Unknown protected cell types: {sorted(unknown_protected)}")
    eligible_indices = np.asarray([
        index for index, cell_type in enumerate(cell_type_names)
        if str(cell_type) not in protected
    ], dtype=int)
    if not len(eligible_indices):
        raise ValueError("At least one cell type must remain eligible for effect injection.")
    ref_disease = disease_levels[0]  # HC
    ref_tissue = tissue_levels[0]  # nif
    other_tissue = tissue_levels[1]  # if
    
    # ------------------------------------
    # Step 1: 确定受影响的细胞集和方向
    # ------------------------------------
    
    # 疾病主效应细胞集 (Disease Main Effect Cells)
    n_disease_main_cts = min(len(eligible_indices), max(1, int(n_celltypes * 0.1)))
    disease_main_cts_indices = rng.choice(
        eligible_indices, size=n_disease_main_cts, replace=False
    )
    # 随机分配方向 (+1 或 -1)
    disease_signs = rng.choice([-1, 1], size=n_disease_main_cts)
    
    # 组织/交互作用效应细胞集 (Tissue/Interaction Effect Cells)
    n_inflamed_cts = min(
        len(eligible_indices),
        0 if inflamed_cell_frac == 0 else max(1, int(n_celltypes * inflamed_cell_frac)),
    )
    inflamed_cts_indices = rng.choice(
        eligible_indices, size=n_inflamed_cts, replace=False
    )
    # NEW: 随机分配方向 (+1 或 -1)
    inflamed_signs = rng.choice([-1, 1], size=n_inflamed_cts)
    
    # --- 2. Disease Main Effects (字典存储) ---
    disease_main_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        random_multiplier = rng.uniform(0.8, 1.2)
        
        # 应用双向效应
        effect_values = disease_effect_size * random_multiplier * disease_signs
        effect_vec[disease_main_cts_indices] = effect_values
        
        disease_main_effects_dict[other_disease] = effect_vec
    
    # --- 3. Tissue Main Effect ---
    tissue_effect_vec = np.zeros(n_celltypes)
    random_multiplier = rng.uniform(0.8, 1.2)  # 同样增加随机性
    
    #  应用双向效应
    tissue_effect_values = tissue_effect_size * random_multiplier * inflamed_signs
    tissue_effect_vec[inflamed_cts_indices] = tissue_effect_values
    
    # --- 4. Disease x Tissue Interaction Effects (字典存储) ---
    interaction_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        random_multiplier = rng.uniform(0.5, 1.5)
        
        # 应用双向效应 (使用与 Tissue Main 效应相同的受影响细胞集和方向，但大小独立)
        interaction_effect_values = interaction_effect_size * random_multiplier * inflamed_signs
        effect_vec[inflamed_cts_indices] = interaction_effect_values
        
        interaction_effects_dict[other_disease] = effect_vec
    
    # --------------------
    # Step 5: 构建 True Effect Table (保持先前修正的参照组和方向判断逻辑)
    # --------------------
    true_effects = []
    
    # 1. Disease Main Effect (Disease vs HC)
    for other_disease, E_vec in disease_main_effects_dict.items():
        for i, ct_name in enumerate(cell_type_names):
            E_disease = E_vec[i]
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'disease',
                'contrast_group': other_disease,
                'contrast_ref': ref_disease,
                'True_Effect': E_disease,
                #  E_disease < 0 时为 ref_greater
                'True_Direction': 'other_greater' if E_disease > 0 else ('ref_greater' if E_disease < 0 else 'None'),
                'True_Significant': True if E_disease != 0 else False
            })
    
    # 2. Tissue Main Effect (if vs nif)
    for i, ct_name in enumerate(cell_type_names):
        E_tissue = tissue_effect_vec[i]
        true_effects.append({
            'cell_type': ct_name,
            'contrast_factor': 'tissue',
            'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,
            'True_Effect': E_tissue,
            # E_tissue < 0 时为 ref_greater
            'True_Direction': 'other_greater' if E_tissue > 0 else ('ref_greater' if E_tissue < 0 else 'None'),
            'True_Significant': True if E_tissue != 0 else False
        })
    
    # 3. Disease x Tissue Interaction
    for other_disease, E_inter_vec in interaction_effects_dict.items():
        E_disease_vec = disease_main_effects_dict[other_disease]
        for i, ct_name in enumerate(cell_type_names):
            E_disease = E_disease_vec[i]
            E_tissue = tissue_effect_vec[i]
            E_interaction = E_inter_vec[i]
            
            # 计算总效应 (Addition 语义)
            total_effect = E_disease + E_tissue + E_interaction
            is_truly_sig = not np.isclose(total_effect, 0.0, atol=1e-12, rtol=0.0)
            
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',
                'contrast_group': f'{other_disease} x {other_tissue}',
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': total_effect,
                # NEW: E_interaction < 0 时为 ref_greater
                'True_Direction': 'other_greater' if total_effect > 0 else (
                    'ref_greater' if total_effect < 0 else 'None'),
                'True_Significant': is_truly_sig
            })
    
    return disease_main_effects_dict, tissue_effect_vec, interaction_effects_dict, pd.DataFrame(true_effects)

