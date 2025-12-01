import anndata
import pandas as pd
from src.utils import ObsEditor
from src.core.base_anndata_ops import subcluster, easy_DEG

adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_5.h5ad")
scRNA_filtered = ObsEditor(adata) # 起个 R 风格的名字以示区分
save_addr="/data/HeLab/bio/IBD_analysis/output/Step08/"
########################################
# PATR 0： 一些单独操作的示例

# 新增一列并进行赋值，并去除缺失行
scRNA_filtered.add_column("disease_type",None)
scRNA_filtered.data.obs["disease_type"] = scRNA_filtered.data.obs["disease"] + "_" + scRNA_filtered.data.obs["tissue"]
scRNA_filtered.drop_missing()

# 修改列名
scRNA_filtered.rename_column("disease_type","disease_tissue")

# 取子集
adata_subset = scRNA_filtered.slice_with_key(obs_key="=Subset_Identity",value="CD4+ Th17",inplace=False)

########################################
# PART 1：将一个 anndata 按照大类进行拆分之后，分类进行降维聚类
# 目的是为了在更精确的范围内捕捉细胞之间相对结构，某种意义上排除“极端值”对细胞亚群聚类的影响
# 先展示简单的方法

# Step 1：将细胞大群拆分
for sub in scRNA_filtered.data.obs.Celltype.unique():
    filename=f"{sub}_subset_analyze.h5ad"
    adata_sub = scRNA_filtered.slice_with_key(obs_key="Celltype",value=sub,inplace=False)
    adata_sub.write_h5ad(filename)

# Step 2：进行简单的分群分析
for sub in scRNA_filtered.data.obs.Celltype.unique():
    res_list = [0.5,1.0,1.5]
    filename=f"{sub}_subset_analyze.h5ad"
    adata_sub = anndata.read_h5ad(filename)
    adata_sub = subcluster(adata_sub, resolutions=res_list) # 对比一下不同的尺度
    for res in res_list:
        leiden_key = f"leiden_res{str(res)}" # leiden_res1_5
        adata_sub = easy_DEG(adata_sub,
                             save_addr=save_addr, filename_prefix=f"{sub}({leiden_key})",
                             obs_key=leiden_key,use_raw=True)


# 对降维聚类的结果进行注释，下面两种是等价的
scRNA_sub = ObsEditor(adata_sub)
# 直接给列表，只能默认识别数字编码的 obs，按从小到大的顺序自动 remap
anno_list = ["Tcm","CD8 Trm-CD16 NK","Tcm","CD4 Tfh"]
scRNA_sub.assign_cluster_identities(annotator = anno_list,
                                    anno_obs_key = "leiden_res0_5",
                                    target_obs_key = "Updated_Identity")
# 给字典，更准确
anno_dict = {"1":"Tcm","2":"CD8 Trm-CD16 NK","3":"Tcm","4":"CD4 Tfh"}
scRNA_sub.assign_cluster_identities(annotator = anno_dict,
                                    anno_obs_key = "leiden_res0_5",
                                    target_obs_key = "Updated_Identity")


# 感觉其中一个身份定义给的不准确，经过仔细检查之后
scRNA_sub.change_one_ident_fast("Updated_Identity",old = "CD8 Trm-CD16 NK",new = "Doublet")

# 现在，把这个更新过的细胞身份（按barcode）更新到原来的对象中
scRNA_filtered.copy_all_ident(adata_from=scRNA_sub.data,
                              from_obs_key="Updated_Identity",
                              to_obs_key="Subset_Identity")

########################################
# PART 2：将一个 anndata 按照大类进行拆分之后，分类进行降维聚类
# 但是这次试用 identity_foucs 对象的工作流程





