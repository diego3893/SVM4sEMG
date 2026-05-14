import csv
import numpy as np
import pandas as pd
import os

def dataconvert(loadfilename, savefilename):
    index = [1, 5] # 更改列号

    csvreader = pd.read_csv(loadfilename)

    data_all = csvreader.to_numpy()
    data_all = data_all[:, index]
    
    np.save(savefilename, data_all)

if __name__ == "__main__":
    input_dir = ["./data/label1", "./data/label0"] 
    
    for dir in input_dir:
        csv_files = [f for f in os.listdir(dir) if f.endswith('.csv')]
    
        if not csv_files:
            print(f"在目录 {dir} 下未找到任何 CSV 文件！")
        else:
            print(f"共找到 {len(csv_files)} 个 CSV 文件，准备开始转换...")
            
            for i, fileName in enumerate(csv_files):
                load_path = os.path.join(dir, fileName)
                
                save_name = f"data{i+1}.npy"
                save_path = os.path.join(dir, save_name)
                
                try:
                    dataconvert(load_path, save_path)
                    print(f"成功: {fileName} -> {save_name}")
                except Exception as e:
                    print(f"跳过文件 {fileName}，发生错误: {e}")

            print("所有转换任务已处理完毕。")