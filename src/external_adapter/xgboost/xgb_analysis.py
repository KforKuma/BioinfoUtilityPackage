# Standard library
import os

# Third-party
import numpy as np
import scanpy as sc
import xgboost as xgb
import seaborn as sns
import shap

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import (StandardScaler, label_binarize)
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, plot_confusion_matrix,
                                 roc_curve, roc_auc_score)
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use("Agg")  # 必须在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt

# Function in this program

from xgb_vis import (plot_tree_importance, plot_confusion_matrix, plot_roc_per_class,
                     plot_fpr_per_class,plot_lodo_confusion_matrix, plot_shap_summary,plot_lodo_stripplots,
                     plot_stability_dendrogram,plot_consensus_dendrogram,plot_stability_clustermap)
from xgb_analysis_utils import (_compute_cm,_read_lodo_outcome,_donor_vectors_from_proba,
                                compute_donor_similarity_matrix,bootstrap_consensus_dendrogram)

#——————————————————————————————————————————————————————————————————————————————————————
def xgb_outcome_analyze(save_path, filename_prefix=None):
    os.makedirs(f"{save_path}/output", exist_ok=True)

    # 读取数据
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    data = np.load(f"{save_path}/{prefix}dataset.npz", allow_pickle=True)

    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"];
    label_mapping = data["label_mapping"].item()

    # 加载模型
    clf = xgb.XGBClassifier()
    clf.load_model(f"{save_path}/{prefix}model.json")

    # 预测
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # 把预测结果单独做一个保存，回头做umap用
    np.save(f"{save_path}/{prefix}y_pred.npy", y_pred)

    # 分类报告
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    filename = f"{save_path}/output/{prefix}classification_result.txt"

    with open(filename, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"[xgb_outcome_analyze] Result is saved at {filename}")

    # 开始做图
    ## 特征重要性
    plot_tree_importance(clf, save_path, filename_prefix)

    ## 混淆矩阵
    confusion_matrix = _compute_cm(y_test,y_pred)
    plot_confusion_matrix(cm_matrix=confusion_matrix,label_mapping=label_mapping,
                          save_path=save_path,filename_prefix=filename_prefix)

    # SHAP分析
    plot_shap_summary(clf,X_test,label_mapping,save_path,filename_prefix)

    # ROC
    plot_roc_per_class(y_test,y_proba,label_mapping,save_path,filename_prefix)

    # F1/Precision/Recall评估
    plot_fpr_per_class(y_test, y_pred, label_mapping, save_path, filename_prefix)

    print("[xgb_outcome_analyze] Finished.\n")


def xgb_outcome_analyze_lodo(save_path,filename_prefix=None):
    '''
    LODO 多每个细胞亚群进行一组LODO的处理

    :param save_path:
    :param filename_prefix:
    :return:
    '''

    results = _read_lodo_outcome(save_path, filename_prefix)

    os.makedirs(f"{save_path}/output",exist_ok=True
                )
    # 绘制箱型图
    print("[xgb_outcome_analyze_lodo] Starting plot_lodo_stripplots ...")
    plot_lodo_stripplots(results, save_path=f"{save_path}/output/combined_boxplot.png")

    # 绘制对比混淆矩阵（Mean|Sigma)
    print("[xgb_outcome_analyze_lodo] Starting plot_lodo_confusion_matrix ...")
    plot_lodo_confusion_matrix(results, save_path=f"{save_path}/output/average_confusion_matrix.png")

    # 映射一个疾病标签用
    donor_labels = results["donor"]
    label_mapping = results['mapping']
    disease_label = [label_mapping.get(lab, lab) for lab in donor_labels]

    print("[xgb_outcome_analyze_lodo] Starting plot_stability_dendrogram ...")
    donor_mat = _donor_vectors_from_proba(results["y_proba"])
    sim = compute_donor_similarity_matrix(donor_mat, metric='cosine')
    plot_stability_dendrogram(sim, disease_label, save_path=f"{save_path}/output/stability_dendro.png")

    print("[xgb_outcome_analyze_lodo] Starting plot_stability_clustermap ...")
    plot_stability_clustermap(sim, disease_label, save_path=f"{save_path}/output/stability_cluster.png")

    print("[xgb_outcome_analyze_lodo] Starting plot plot_consensus_dendrogram ...")
    consensus_tree, branch_supports = bootstrap_consensus_dendrogram(sim_matrix=sim, n_bootstrap=100,
                                                                     method="average", support_threshold=0.5)
    np.fill_diagonal(sim, 0)  # 直接修改 sim
    D_condensed = squareform(sim)  # 对角设为0，把方阵转换成一维距离向量
    Z_real = linkage(D_condensed, method='average')  # 或 'complete', 'ward' 等
    plot_consensus_dendrogram(Z_real, disease_label, branch_supports,
                              save_path=f"{save_path}/output/consensual_tree.png")

    print("[xgb_outcome_analyze_lodo] Finished.\n")



