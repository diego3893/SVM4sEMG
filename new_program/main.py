import os
import pandas as pd
import numpy as np
from scipy import signal
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 屏蔽版本兼容警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 导入特征提取函数
try:
    from features import Get_EMG_Feature
except ImportError:
    print("错误：在当前目录下找不到 features.py，请检查文件位置。")

FS = 2000
WINDOW_LENGTH = 100
WINDOW_MOVE_STEP = 20
DATA_DIR = "./data"

def data_filter(data, fs=FS):
    # 20-200Hz 带通滤波器
    b, a = signal.butter(4, [2 * 20 / fs, 2 * 200 / fs], "bandpass")
    filtered = signal.filtfilt(b, a, data, axis=0)
    # 这一步可以去除某些晃动导致的物理噪声
    # 50Hz 陷波滤波器 (针对工频干扰)
    b1, a1 = signal.butter(4, [2 * 48 / fs, 2 * 52 / fs], "bandstop")
    filtered = signal.filtfilt(b1, a1, filtered, axis=0)
    return filtered
    # 去除交流电50Hz的干扰
    # 这是数据预处理，过滤噪声

def process_single_csv(file_path):
    # 读取csv
    try:
        csv_file = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return None
    
    emg_target_cols = [col for col in csv_file.columns if "EMG" in str(col).upper()]
    
    if not emg_target_cols:
        # 如果没有匹配到目标列，且总列数大于6，直接提取第2列和第6列
        if csv_file.shape[1] >= 6:
            return csv_file.iloc[:, [1, 5]].values
        return None
    
    return csv_file[emg_target_cols].values

def extract_features_from_data(emg_data):
    # 删除前200ms
    remove_points = int(200 * FS / 1000)
    if emg_data.shape[0] > remove_points:
        emg_data = emg_data[remove_points:, :] # 切片
    else:
        return np.array([]) # 不到200ms，作废

    filtered_data = data_filter(emg_data)
    
    # 将时间毫秒转换为采样点数
    win_points = int(WINDOW_LENGTH * FS / 1000)
    step_points = int(WINDOW_MOVE_STEP * FS / 1000)
    
    num_windows = int((filtered_data.shape[0] - win_points) / step_points) + 1
    
    feature_list = []
    for i in range(num_windows):
        window_data = filtered_data[i * step_points : i * step_points + win_points]
        # 用feat1得出四种特征
        feat = Get_EMG_Feature("feat1", window_data, window=win_points, step=step_points)
        feature_list.append(feat.flatten())
        
    return np.array(feature_list)

def main():
    all_features = []
    all_labels = []

    if not os.path.exists(DATA_DIR):
        print(f"未找到数据文件夹")
        return

    folder_label_map = { # 子文件夹和标签映射关系
        "label0": 0,
        "label1": 1
    }

    total_files_processed = 0

    for folder_name, label in folder_label_map.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"找不到子文件夹 '{folder_path}'，已跳过。")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')] # 找所有csv
        
        if not files:
            print(f"在 '{folder_path}' 中没有找到任何 CSV 文件。")
            continue

        print(f"开始处理 '{folder_path}' 中的 {len(files)} 个文件 (标签: {label})...")

        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            raw_emg = process_single_csv(file_path)

            if raw_emg is None or len(raw_emg) < (WINDOW_LENGTH * FS / 1000):
                print(f"  -> 跳过文件: {filename} (数据为空或长度不足)")
                continue
            
            print(f"  -> 正在处理: {filename}")
            
            feats = extract_features_from_data(raw_emg)
            all_features.append(feats)
            all_labels.append(np.full(feats.shape[0], label))
            total_files_processed += 1

    if not all_features:
        print("错误：未提取到任何有效特征。请检查 CSV 数据内容。")
        return

    # 合并所有文件夹的数据
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    # 划分 80% 训练，20% 测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n--- 数据集概况 ---")
    print(f"成功读取的有效文件数: {total_files_processed}")
    print(f"总窗口数 (样本量): {X.shape[0]}")
    print(f"特征维度: {X.shape[1]}")

    print("\n--- 提取出的特征原始数值观察（归一化前） ---")
    num_channels = X.shape[1] // 4  
    
    mav_val = X[0, 0]                            # f1: MAV
    mavslp_val = X[0, num_channels]              # f2: MAVSLP
    zc_val = X[0, num_channels * 2]              # f3: ZC
    wl_val = X[0, num_channels * 3]              # f5: WL

    print(f"1. MAV    (平均绝对值) : {mav_val:12.6f}")
    print(f"2. MAVSLP (均值斜率)   : {mavslp_val:12.6f}")
    print(f"3. ZC     (过零率)     : {zc_val:12.6f}")
    print(f"4. WL     (波形长度)   : {wl_val:12.6f}")
    print("============================================\n")

    scaler = StandardScaler()
    # 归一化
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n--- 提取出的特征数值观察（归一化后） ---")
    
    mav_val_norm = X_train[0, 0]
    mavslp_val_norm = X_train[0, num_channels]
    zc_val_norm = X_train[0, num_channels * 2]
    wl_val_norm = X_train[0, num_channels * 3]

    print(f"1. MAV    (平均绝对值) : {mav_val_norm:12.6f}")
    print(f"2. MAVSLP (均值斜率)   : {mavslp_val_norm:12.6f}")
    print(f"3. ZC     (过零率)     : {zc_val_norm:12.6f}")
    print(f"4. WL     (波形长度)   : {wl_val_norm:12.6f}")
    print("============================================\n")
    
    # 训练 SVM 分类器
    print("\n正在训练模型，请稍候...")

    param_grid = { # 需要枚举的训练参数
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"找到的最佳参数组合: {grid.best_params_}")

    # 结果评估
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred) # 运用测试集进行评估
    print(f"--- 实验分析结果 ---")
    print(f"分类识别准确率: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()


# 73.15，82.81，84.53，85.67
# 初始训练准确度在73.15
# 尝试删除每个csv的前200ms（前400个数据点），准度到82.81
# 对特征做归一化，准度为84.53
# 尝试自动调整参数（枚举参数），准度为85.67