from __future__ import annotations
import warnings
import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd



from src.utils.hier_logger import logged
from src.stats.simulation.design import build_simulation_metadata, validate_factor_levels
from src.stats.simulation.truth_refine import attach_population_effects, refine_ground_truth_by_observation
logger = logging.getLogger(__name__)

# -----------------------
# 生成模拟数据：“真实数据 resampling” 模拟
# 相对最公正
# -----------------------
@logged
def simulate_CLR_resample_data(
        count_df,
        n_donors=20,
        n_samples_per_donor=4,
        n_celltypes=30,  # 新增参数，默认设为你想要的 100
        disease_effect_size=0.5,
        tissue_effect_size=0.8,
        interaction_effect_size=0.5,
        inflamed_cell_frac=0.15,
        donor_noise_sd=0.2,
        sample_noise_sd=0.1,
        disease_levels=("HC", "CD", "UC"),
        tissue_levels=("nif", "if"),
        pseudocount=1.0,
        random_state=1234,
        assignment_strategy="balanced",
        protected_cell_types: Sequence[str] = (),
        population_null_tolerance: float = 1e-12,
        population_reference_cell_type: str | None = None,
):
    """基于真实样本 CLR 背景重采样生成模拟丰度数据。

    该模拟器先从真实参考组 ``ref_disease x ref_tissue`` 中抽取 baseline CLR logits，
    再叠加 donor/sample 噪声和注入效应，最后按真实 sample 的测序深度分布进行
    multinomial 采样。它通常更贴近真实数据结构，适合作为相对保守的模拟方案。

    Args:
        count_df: 真实长表 count 数据，至少包含 ``sample_id``、``donor_id``、
            ``disease``、``tissue``、``cell_type`` 和 ``count``。
        n_donors: 模拟 donor 数量。
        n_samples_per_donor: 每个 donor 的 sample 数量。
        n_celltypes: 模拟 cell subtype/subpopulation 数量；可从真实亚群中抽样或重复抽样。
        disease_effect_size: disease 主效应大小。
        tissue_effect_size: tissue 主效应大小。
        interaction_effect_size: 交互效应大小。
        inflamed_cell_frac: 受 tissue/interaction 影响的亚群比例。
        donor_noise_sd: donor 层面 CLR shift 标准差。
        sample_noise_sd: sample 层面 CLR 噪声标准差。
        disease_levels: disease 水平，首个元素作为参考组。
        tissue_levels: tissue 水平，首个元素作为参考组。
        pseudocount: 从真实 count 转 CLR logits 前加入的伪计数。
        random_state: 随机种子。
        assignment_strategy: ``balanced`` 为默认的分层随机平衡设计；``random``
            保留历史独立随机分配行为。
        protected_cell_types: 不得承载 latent injection 的预注册 cell type。
        population_null_tolerance: 判定精确 population effect 数值零的显式容差。
        population_reference_cell_type: population log-ratio estimand 的预注册 reference。

    Returns:
        ``(df_long, df_true_refined)``，分别为模拟长表和按实际观察 LFC 修正后的
        ground truth。

    Example:
        >>> df_sim, df_truth = simulate_CLR_resample_data(
        ...     count_df=real_count_df,
        ...     n_donors=20,
        ...     n_samples_per_donor=4,
        ...     n_celltypes=30,
        ...     disease_levels=("HC", "CD", "UC"),
        ...     tissue_levels=("nif", "if"),
        ...     random_state=42,
        ... )
        >>> df_sim[["sample_id", "cell_type", "count", "prop"]].head()
        # 与真实参考样本背景相近的模拟组成数据。
    """
    required_cols = {"sample_id", "donor_id", "disease", "tissue", "cell_type", "count"}
    missing_cols = required_cols - set(count_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    if not isinstance(n_celltypes, int) or isinstance(n_celltypes, bool) or n_celltypes <= 1:
        raise ValueError("`n_celltypes` must be an integer greater than 1.")
    disease_levels, tissue_levels = validate_factor_levels(disease_levels, tissue_levels)
    numeric_parameters = np.asarray([
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, donor_noise_sd, sample_noise_sd, pseudocount,
        population_null_tolerance,
    ], dtype=float)
    if not np.isfinite(numeric_parameters).all():
        raise ValueError("Simulation numeric parameters must be finite.")
    if not 0 <= inflamed_cell_frac <= 1:
        raise ValueError("`inflamed_cell_frac` must be between 0 and 1.")
    if min(disease_effect_size, tissue_effect_size, interaction_effect_size) < 0:
        raise ValueError("Effect-size parameters are non-negative magnitudes.")
    if min(donor_noise_sd, sample_noise_sd) < 0:
        raise ValueError("Noise scale parameters must be non-negative.")
    if pseudocount <= 0:
        raise ValueError("`pseudocount` must be greater than 0.")

    counts_numeric = pd.to_numeric(count_df["count"], errors="coerce")
    if counts_numeric.isna().any() or ~np.isfinite(counts_numeric).all() or counts_numeric.lt(0).any():
        raise ValueError("Input counts must be finite and non-negative.")
    if not np.equal(counts_numeric, np.floor(counts_numeric)).all():
        raise ValueError("Input counts must be integer-valued.")
    count_df = count_df.copy()
    count_df["count"] = counts_numeric

    rng = np.random.default_rng(random_state)
    n_sim_samples = n_donors * n_samples_per_donor
    
    # ---------------------------
    # Step 1: 数据提取与基线构建
    # ---------------------------
    metadata = count_df[['sample_id', 'donor_id', 'disease', 'tissue']].drop_duplicates()
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Input donor/disease/tissue metadata must be sample-constant.")
    metadata_map = metadata.set_index('sample_id')
    sample_totals = count_df.groupby('sample_id')['count'].sum()
    if sample_totals.le(0).any():
        raise ValueError("Every source sample must have a positive total count.")
    
    df_counts_wide = (
        count_df.groupby(['sample_id', 'cell_type'])['count']
        .sum()
        .unstack(fill_value=0)
    )
    
    # ---------------------------
    # Step 2: 关键修改 - 细胞维度重塑
    # ---------------------------
    cell_types_original = df_counts_wide.columns.tolist()
    n_orig = len(cell_types_original)
    
    # 根据目标 n_celltypes 采样或补全原始索引
    if n_celltypes <= n_orig:
        selected_orig_indices = rng.choice(n_orig, size=n_celltypes, replace=False)
    else:
        # 如果需要的比原始多，则允许重复采样原始细胞背景
        selected_orig_indices = np.concatenate([
            np.arange(n_orig),
            rng.choice(n_orig, size=n_celltypes - n_orig, replace=True)
        ])
    
    # 构建新的虚拟细胞名称列表 [CT1, CT2, ..., CTn]
    sim_cell_names = [f"CT{i + 1}" for i in range(n_celltypes)]
    if (
        population_reference_cell_type is not None
        and str(population_reference_cell_type) not in set(map(str, protected_cell_types))
    ):
        raise ValueError("The population reference cell type must also be protected from injection.")
    
    # 获取基线样本池
    ref_disease = disease_levels[0]
    ref_tissue = tissue_levels[0]
    baseline_sample_ids = metadata_map[
        (metadata_map['disease'] == ref_disease) & (metadata_map['tissue'] == ref_tissue)
        ].index
    
    if len(baseline_sample_ids) == 0:
        raise ValueError(f"Baseline sample pool is empty. Expected disease: '{ref_disease}' and tissue: '{ref_tissue}'.")
    
    # 提取 count 矩阵并根据 selected_orig_indices 进行切片/重组
    # 注意：这里我们提取了指定维度的 Logits 背景
    counts_baseline = df_counts_wide.loc[baseline_sample_ids].values[:, selected_orig_indices] + pseudocount
    log_counts = np.log(counts_baseline)
    clr_logits_baseline = log_counts - np.mean(log_counts, axis=1, keepdims=True)
    
    # ---------------------------
    # Step 3: 设计效应向量 (使用新的 sim_cell_names)
    # ---------------------------
    disease_main_effects_dict, tissue_effect, interaction_effects_dict, df_true_effect = build_CLR_effects_and_table(
        cell_types=sim_cell_names,
        disease_levels=disease_levels,
        tissue_levels=tissue_levels,
        disease_effect_size=disease_effect_size,
        interaction_effect_size=interaction_effect_size,
        tissue_effect_size=tissue_effect_size,
        inflamed_cell_frac=inflamed_cell_frac,
        rng=rng,
        protected_cell_types=protected_cell_types,
    )
    
    # ---------------------------
    # Step 4: 层次化模拟 (与之前逻辑一致，但维度已变为 n_celltypes)
    # ---------------------------
    df_sim_meta = build_simulation_metadata(
        n_donors=n_donors,
        n_samples_per_donor=n_samples_per_donor,
        disease_levels=disease_levels,
        tissue_levels=tissue_levels,
        rng=rng,
        assignment_strategy=assignment_strategy,
    )
    donor_shifts = {
        donor: rng.normal(0, donor_noise_sd, n_celltypes)
        for donor in df_sim_meta["donor_id"].unique()
    }
    simulated_logits = []
    for sample in df_sim_meta.itertuples(index=False):
        idx_resample = rng.integers(0, len(clr_logits_baseline))
        clr_logit_sim = clr_logits_baseline[idx_resample].copy()
        clr_logit_sim += donor_shifts[sample.donor_id]
        if sample.disease != ref_disease:
            clr_logit_sim += disease_main_effects_dict[sample.disease]
        if sample.tissue != ref_tissue:
            clr_logit_sim += tissue_effect
        if sample.disease != ref_disease and sample.tissue != ref_tissue:
            clr_logit_sim += interaction_effects_dict[sample.disease]
        clr_logit_sim += rng.normal(0, sample_noise_sd, n_celltypes)
        simulated_logits.append(clr_logit_sim)
    df_sim_meta["clr_logit_sim"] = simulated_logits
    
    # ---------------------------
    # Step 5: 生成 Count
    # ---------------------------
    logits_matrix = np.vstack(df_sim_meta['clr_logit_sim'].values)
    logits_matrix = np.clip(logits_matrix, -700, 700)
    exp_logits = np.exp(logits_matrix)
    proportions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # multinomial 需要整数测序深度；真实数据若被读成 float，这里统一兜底转换。
    sim_depths = rng.choice(sample_totals.values, size=n_sim_samples, replace=True).astype(int)
    counts_matrix = np.array([
        rng.multinomial(n=sim_depths[i], pvals=proportions[i])
        for i in range(n_sim_samples)
    ])
    
    # ---------------------------
    # Step 6: 内存高效构建长表
    # ---------------------------
    df_sim_long = df_sim_meta.iloc[np.repeat(np.arange(n_sim_samples), n_celltypes)].copy()
    df_sim_long.drop(columns='clr_logit_sim', inplace=True)
    
    # 填充 cell_type 编号
    df_sim_long['cell_type'] = np.tile(sim_cell_names, n_sim_samples)
    df_sim_long['count'] = counts_matrix.flatten()
    
    df_sim_long['total_count'] = df_sim_long.groupby('sample_id')['count'].transform('sum')
    df_sim_long['prop'] = df_sim_long['count'] / df_sim_long['total_count']
    df_true_effect = attach_population_effects(
        df_true_effect,
        baseline_logits=clr_logits_baseline,
        cell_types=sim_cell_names,
        disease_effects=disease_main_effects_dict,
        tissue_effect=tissue_effect,
        interaction_effects=interaction_effects_dict,
        population_null_tolerance=population_null_tolerance,
        population_reference_cell_type=population_reference_cell_type,
    )
    df_long = df_sim_long.reset_index(drop=True)
    df_true_refined = refine_ground_truth_by_observation(
        df_long, df_true_effect, injected_effect_scale="clr_latent_effect"
    )
    return df_long, df_true_refined

@logged
def build_CLR_effects_and_table(
        cell_types, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng, protected_cell_types: Sequence[str] = ()
):
    """构建 CLR resampling 模拟所需的效应向量和 ground truth 表。

    当前逻辑保留三个语义：不同 disease 可影响不同亚群；interaction 与 tissue
    主效应的影响亚群解耦；显著性判定基于主效应和交互效应的叠加结果。

    Args:
        cell_types: 模拟 cell subtype/subpopulation 名称列表。
        disease_levels: disease 水平，首个元素作为参考组。
        tissue_levels: tissue 水平，首个元素作为参考组。
        disease_effect_size: disease 主效应大小。
        tissue_effect_size: tissue 主效应大小。
        interaction_effect_size: interaction 效应大小。
        inflamed_cell_frac: 受 tissue/interaction 影响的亚群比例。
        rng: ``numpy.random.Generator`` 实例。

    Returns:
        ``(disease_main_effects_dict, tissue_effect, interaction_effects_dict,
        df_true_effect)``。

    Example:
        >>> rng = np.random.default_rng(3)
        >>> disease_eff, tissue_eff, inter_eff, truth = build_CLR_effects_and_table(
        ...     ["CT1", "CT2", "CT3"],
        ...     ("HC", "CD"),
        ...     ("nif", "if"),
        ...     0.5,
        ...     0.8,
        ...     0.4,
        ...     0.2,
        ...     rng,
        ... )
        >>> truth.query("True_Significant").head()
        # 返回每个真实注入差异的方向和效应大小。
    """
    n_celltypes = len(cell_types)
    if n_celltypes == 0:
        raise ValueError("`cell_types` must not be empty.")
    protected = set(map(str, protected_cell_types))
    unknown_protected = protected - set(map(str, cell_types))
    if unknown_protected:
        raise ValueError(f"Unknown protected cell types: {sorted(unknown_protected)}")
    eligible_indices = np.asarray([
        index for index, cell_type in enumerate(cell_types)
        if str(cell_type) not in protected
    ], dtype=int)
    if not len(eligible_indices):
        raise ValueError("At least one cell type must remain eligible for effect injection.")
    if len(tissue_levels) < 2:
        raise ValueError("`tissue_levels` must contain at least two levels.")
    ref_disease = disease_levels[0]  # HC
    ref_tissue = tissue_levels[0]  # nif
    other_tissue = tissue_levels[1]  # if
    
    n_disease_main_cts = min(len(eligible_indices), max(1, int(n_celltypes * 0.1)))
    n_inflamed_cts = min(
        len(eligible_indices),
        0 if inflamed_cell_frac == 0 else max(1, int(n_celltypes * inflamed_cell_frac)),
    )
    
    # --- 1. 疾病主效应 (每个疾病独立采样受影响细胞) ---
    disease_main_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        # 只有在 size > 0 时才采样和赋值
        if disease_effect_size > 0:
            indices = rng.choice(eligible_indices, size=n_disease_main_cts, replace=False)
            signs = rng.choice([-1, 1], size=n_disease_main_cts)
            random_multiplier = rng.uniform(0.8, 1.2)
            effect_vec[indices] = disease_effect_size * random_multiplier * signs
        disease_main_effects_dict[other_disease] = effect_vec
    
    # --- 2. 组织主效应 ---
    tissue_effect = np.zeros(n_celltypes)
    if tissue_effect_size > 0:
        inflamed_cts_indices = rng.choice(
            eligible_indices, size=n_inflamed_cts, replace=False
        )
        inflamed_signs = rng.choice([-1, 1], size=n_inflamed_cts)
        random_multiplier = rng.uniform(0.8, 1.2)
        tissue_effect[inflamed_cts_indices] = tissue_effect_size * random_multiplier * inflamed_signs
    
    # --- 3. 交互作用效应 (独立采样，不强制与组织效应细胞重合) ---
    interaction_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        if interaction_effect_size > 0:
            # 独立采样交互项影响的细胞，数量仍由 inflamed_cell_frac 决定
            inter_indices = rng.choice(eligible_indices, size=n_inflamed_cts, replace=False)
            inter_signs = rng.choice([-1, 1], size=n_inflamed_cts)
            random_multiplier = rng.uniform(0.5, 1.5)
            effect_vec[inter_indices] = interaction_effect_size * random_multiplier * inter_signs
        interaction_effects_dict[other_disease] = effect_vec
    
    # --------------------
    # 构建 True Effect Table
    # --------------------
    true_effects = []
    
    # 1. Disease Main Effect
    for other_disease, E_vec in disease_main_effects_dict.items():
        for i, ct_name in enumerate(cell_types):
            val = E_vec[i]
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'disease',
                'contrast_group': other_disease,
                'contrast_ref': ref_disease,
                'True_Effect': val,
                'True_Direction': 'other_greater' if val > 0 else ('ref_greater' if val < 0 else 'None'),
                'True_Significant': True if val != 0 else False
            })
    
    # 2. Tissue Main Effect
    for i, ct_name in enumerate(cell_types):
        val = tissue_effect[i]
        true_effects.append({
            'cell_type': ct_name,
            'contrast_factor': 'tissue',
            'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,
            'True_Effect': val,
            'True_Direction': 'other_greater' if val > 0 else ('ref_greater' if val < 0 else 'None'),
            'True_Significant': True if val != 0 else False
        })
    
    # 3. Disease x Tissue Interaction
    for other_disease, E_inter_vec in interaction_effects_dict.items():
        # 获取疾病主效应
        E_disease_vec = disease_main_effects_dict[other_disease]
        for i, ct_name in enumerate(cell_types):
            val_disease = E_disease_vec[i]
            val_tissue = tissue_effect[i]
            val_inter = E_inter_vec[i]
            
            # 叠加
            total_val = val_disease + val_tissue + val_inter
            is_truly_sig = not np.isclose(total_val, 0.0, atol=1e-12, rtol=0.0)
            
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',
                'contrast_group': f'{other_disease} x {other_tissue}',
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': total_val,
                'True_Direction': 'other_greater' if total_val > 0 else ('ref_greater' if total_val < 0 else 'None'),
                'True_Significant': is_truly_sig
            })
    
    return disease_main_effects_dict, tissue_effect, interaction_effects_dict, pd.DataFrame(true_effects)
