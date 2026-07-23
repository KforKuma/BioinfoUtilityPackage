from .Tri_anchor import (
    TriAnchorAdapter,
    TriAnchorRule,
    assess_anchor_compatibility,
    build_tri_anchor_result,
    combine_anchor_evidence,
    load_tri_anchor_rule,
    prepare_anchor_inputs,
    run_Meta_Ensemble,
    run_Meta_Ensemble_adaptive,
)
# from .sccoda import run_scCODA

# 集中导出 meta engine。scCODA 依赖较重且当前不作为默认集合方法导出。
__all__ = [
    "TriAnchorAdapter", "TriAnchorRule", "prepare_anchor_inputs",
    "assess_anchor_compatibility", "combine_anchor_evidence",
    "build_tri_anchor_result", "load_tri_anchor_rule",
    "run_Meta_Ensemble", "run_Meta_Ensemble_adaptive"
    # 'run_scCODA'
]
