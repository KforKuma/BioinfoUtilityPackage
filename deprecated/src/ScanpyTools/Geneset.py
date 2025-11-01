import pandas as pd
# 迁移完成

class Geneset():
    def __init__(self, marker_file):
        self.data = {}
        excel_data = pd.ExcelFile(marker_file)
        for sheet in excel_data.sheet_names:
            print(f"Reading sheet of {sheet}")
            df = excel_data.parse(sheet)

            # 过滤掉 gene_set 列为空的行，并复制干净数据
            df = df.dropna(subset=['gene_set']).copy()

            # 去除方括号并拆分基因集
            try:
                df['gene_set'] = (
                    df['gene_set']
                    .astype(str)  # 确保是字符串
                    .str.strip('[]')  # 去除方括号
                    .str.split(',')  # 拆分成列表
                )

                # 清理引号与空格
                df['gene_set'] = df['gene_set'].apply(
                    lambda gene_list: [
                        gene.strip().strip("'").strip('"') for gene in gene_list
                    ]
                )
            except Exception as e:
                print(f"Error while parsing gene_set in sheet '{sheet}': {e}")
                raise e

            self.data[sheet] = df

    def disable_cell_type(self, marker_sheet, cell_type):
        # 将指定的 cell_type 标记为禁用
        row_mask = self.data[marker_sheet]['cell_type'] == cell_type
        if row_mask.any():
            self.data[marker_sheet].loc[row_mask, 'disabled'] = True
            print(f"Cell type '{cell_type}' has been disabled.")
        else:
            print(f"Cell type '{cell_type}' not found in sheet '{marker_sheet}'.")

    def enable_cell_type(self, marker_sheet, cell_type):
        # 重新启用指定的 cell_type
        row_mask = self.data[marker_sheet]['cell_type'] == cell_type
        if row_mask.any():
            self.data[marker_sheet].loc[row_mask, 'disabled'] = False
            print(f"Cell type '{cell_type}' has been enabled.")
        else:
            print(f"Cell type '{cell_type}' not found in sheet '{marker_sheet}'.")

    def get_gene_dict(self, marker_sheet, celltype_list="", facet_split=False):
        # 过滤掉被禁用的 cell_type
        df = self.data[marker_sheet][self.data[marker_sheet]['disabled'] == False]

        # 辅助函数：构建字典并根据 celltype_list 进行过滤
        def build_gene_dict(dataframe):
            gene_dict = dict(zip(dataframe['cell_type'], dataframe['gene_set']))
            if celltype_list:
                gene_dict = {key: gene_dict[key] for key in celltype_list if key in gene_dict}
            return gene_dict

        if facet_split:
            # 按 'facet' 列进行分组，并对每组应用 build_gene_dict
            gene_dicts = {facet: build_gene_dict(sub_df) for facet, sub_df in df.groupby('facet')}
            return gene_dicts
        else:
            # 不分组，直接构建 gene_dict
            return build_gene_dict(df)

    def update_gene_dict(self, gene_dict, marker_sheet):
        # 遍历 gene_dict，更新相应的基因集
        for key, new_geneset in gene_dict.items():
            # 使用布尔索引直接筛选行
            row_mask = self.data[marker_sheet]['cell_type'] == key

            if row_mask.any():  # 如果找到对应的 cell_type
                original_geneset = self.data[marker_sheet].loc[row_mask, 'gene_set'].iloc[0]
                if set(original_geneset) == set(new_geneset):
                    print(f"{key} gene_set has not changed.")
                else:
                    print(f"{key} gene_set has changed.\n")
                    print("The original gene set is:\n", original_geneset)
                    print("The new gene set is:\n", new_geneset)
                    # 输入验证
                    while True:
                        gene_set_select = input("Please make a choice, enter 'o' for original or 'n' for new: ").lower()
                        if gene_set_select == 'o':
                            break
                        elif gene_set_select == 'n':
                            self.data[marker_sheet].loc[row_mask, 'gene_set'] = new_geneset
                            break
                        else:
                            print("Invalid input, please enter 'o' or 'n'.")
            else:
                # 如果没有找到相应的 cell_type，则添加新行
                new_row = {'cell_type': key, 'gene_set': new_geneset, 'remark': None}
                self.data[marker_sheet] = pd.concat([self.data[marker_sheet], pd.DataFrame([new_row])],
                                                    ignore_index=True)

    def save(self, marker_file_save):
        with pd.ExcelWriter(marker_file_save, engine='openpyxl') as writer:
            for sheet_name, df in self.data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)