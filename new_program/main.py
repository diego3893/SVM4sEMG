import os
import pandas as pd
import numpy as np
from scipy import signal
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# 屏蔽 Anaconda 环境中常见的版本兼容警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 导入特征提取函数 (请确保 features.py 在当前脚本同级目录下)
try:
    from features import Get_EMG_Feature
except ImportError:
    print("错误：在当前目录下找不到 features.py，请检查文件位置。")

# ================= 配置区 =================
FS = 2000          # Delsys 默认采样率 2000Hz
WIN_LEN = 100      # 窗口长度 (ms)
WIN_MOVE = 20      # 滑动步长 (ms)
DATA_DIR = "./my_data"  # 你的数据主目录
# ==========================================

def data_filter(data, fs=FS):
    """带通滤波 + 50Hz 陷波滤波"""
    # 20-200Hz 带通滤波器
    b, a = signal.butter(4, [2 * 20 / fs, 2 * 200 / fs], "bandpass")
    filtered = signal.filtfilt(b, a, data, axis=0)
    
    # 50Hz 陷波滤波器 (针对工频干扰)
    b1, a1 = signal.butter(4, [2 * 48 / fs, 2 * 52 / fs], "bandstop")
    filtered = signal.filtfilt(b1, a1, filtered, axis=0)
    return filtered

def process_single_csv(file_path):
    """读取 CSV 并自动提取 EMG 通道数据"""
    # 使用 pandas 读取 CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return None
    
    # 自动搜索列名中包含 "EMG" 的所有列
    emg_cols = [col for col in df.columns if "EMG" in str(col).upper()]
    
    if not emg_cols:
        # 如果列名中不含 'EMG'，则根据 Delsys 默认格式尝试提取特定列
        # 常见位置为第 2 和第 6 列（索引 1, 5），若仍不对请手动修改
        if df.shape[1] >= 6:
            return df.iloc[:, [1, 5]].values
        return None
    
    return df[emg_cols].values

def extract_features_from_data(emg_data):
    # 【新增】：删除前 200 毫秒的数据
    # 因为采样率 FS = 2000，200ms 对应的就是 200 * (2000/1000) = 400 个采样点
    remove_points = int(200 * FS / 1000)
    if emg_data.shape[0] > remove_points:
        emg_data = emg_data[remove_points:, :] # 切片：只保留 200ms 之后的数据
    else:
        return np.array([]) # 如果整个数据还不到 200ms，直接作废

    filtered_data = data_filter(emg_data)
    """滑动窗口并调用 features.py 提取特征"""
    filtered_data = data_filter(emg_data)
    
    # 将时间毫秒转换为采样点数
    win_points = int(WIN_LEN * FS / 1000)
    step_points = int(WIN_MOVE * FS / 1000)
    
    num_windows = int((filtered_data.shape[0] - win_points) / step_points) + 1
    
    feature_list = []
    for i in range(num_windows):
        window_data = filtered_data[i * step_points : i * step_points + win_points]
        # 调用提供的 Get_EMG_Feature 提取时域特征组合
        feat = Get_EMG_Feature("feat1", window_data, window=win_points, step=step_points)
        feature_list.append(feat.flatten())
        
    return np.array(feature_list)

def main():
    all_features = []
    all_labels = []

    # 检查主文件夹是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误：在当前目录下找不到名为 '{DATA_DIR}' 的主文件夹，请先创建它。")
        return

    # 定义子文件夹与标签的映射关系
    folder_label_map = {
        "label0": 0,
        "label1": 1
    }

    total_files_processed = 0

    # 遍历每个子文件夹
    for folder_name, label in folder_label_map.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        
        # 检查子文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"警告：找不到子文件夹 '{folder_path}'，已跳过。")
            continue
            
        # 获取该子文件夹下所有的 CSV 文件
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not files:
            print(f"提示：在 '{folder_path}' 文件夹中没有找到任何 CSV 文件。")
            continue

        print(f"开始处理 '{folder_path}' 中的 {len(files)} 个文件 (标签: {label})...")

        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            raw_emg = process_single_csv(file_path)
            
            # 简单检查数据长度是否足够一个窗口
            if raw_emg is None or len(raw_emg) < (WIN_LEN * FS / 1000):
                print(f"  -> 跳过文件: {filename} (数据为空或长度不足)")
                continue
            
            print(f"  -> 正在处理: {filename}")
            
            # 提取特征
            feats = extract_features_from_data(raw_emg)
            all_features.append(feats)
            # 生成对应长度的标签数组
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
    
    # 训练 SVM 分类器
    print("\n正在训练模型，请稍候...")
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale') # 这里顺手把 gamma 改成了更稳定的 'scale'
    clf.fit(X_train, y_train)

    # 结果评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- 实验分析结果 ---")
    print(f"分类识别准确率: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()