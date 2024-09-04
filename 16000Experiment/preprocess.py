import numpy as np
import pandas as pd

def preprocess_data(input_file, output_file):
    # 读取数据
    df_merged = pd.read_csv(input_file, header=None)
    
    # 删除全部为0或者全部为NaN的列
    df_merged = df_merged.loc[:, ~(df_merged.isnull() | (df_merged == 0)).all(axis=0)]
    
    # 删除数据全部一样的列
    df_merged = df_merged.loc[:, df_merged.nunique() > 1]
    
    # 保存预处理后的数据
    df_merged.to_csv(output_file, index=False, header=False)
    print(f"Data preprocessed successfully, remaining columns: {df_merged.shape[1]}")
    return df_merged

if __name__ == "__main__":
    input_file = "merged_configuration_all.csv"
    output_file = "merged_configuration_all_processed.csv"
    preprocess_data(input_file, output_file)
