import numpy as np
from scipy import signal
from features import Get_EMG_Feature
from sklearn import svm
from sklearn.model_selection import train_test_split
import os  # 新增：用于处理文件路径

# ... (data_filter 函数保持不变) ...
def data_filter(data):
    fs = 2000
    low_fs = 20
    high_fs = 200
    b, a = signal.butter(4, [2 * low_fs / fs, 2 * high_fs / fs], "bandpass")
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    low_fs1 = 48
    high_fs1 = 52
    b1, a1 = signal.butter(4, [2 * low_fs1 / fs, 2 * high_fs1 / fs], "bandstop")
    filtered_data = signal.filtfilt(b1, a1, filtered_data, axis=0)
    return filtered_data

def data_process(filename, label):
    data = np.load(filename)
    
    if data.shape[0] > 400:
        data = data[400:, :]
    else:
        data = data[0:0, :]

    filtered_data = data_filter(data)
    win_len = 100
    win_move = 20
    
    if filtered_data.shape[0] < win_len:
        return None, None
        
    num_all = int((filtered_data.shape[0] - win_len) / win_move) + 1
    data_ = np.zeros((num_all, win_len, filtered_data.shape[1]))
    for i in range(num_all):
        data_[i, :, :] = filtered_data[i * win_move : i * win_move + win_len]
    
    data_feature = []
    for i in range(num_all):
        data_feature.append(Get_EMG_Feature("feat1", data_[i, :, :]))
    
    data_feature = np.array(data_feature)
    if data_feature.ndim > 2:
        data_feature = data_feature.squeeze()
        
    data_label = np.ones((num_all, 1)) * label
    return data_feature, data_label


def process_label_directory(directory, label):
    all_features = []
    all_labels = []
    
    if not os.path.exists(directory):
        print(f"警告：目录 {directory} 不存在")
        return None, None

    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    for f in files:
        file_path = os.path.join(directory, f)
        try:
            feat, lbl = data_process(file_path, label)
            if feat is not None and lbl is not None:
                all_features.append(feat)
                all_labels.append(lbl)
                print(f"已处理: {f} (标签: {label})")
        except Exception as e:
            print(f"处理文件 {f} 时出错: {e}")
            
    if not all_features:
        return None, None
        
    return np.vstack(all_features), np.vstack(all_labels)

def SVM_experiment():
    feat0, lbl0 = process_label_directory("./data/label0", 0)
    feat1, lbl1 = process_label_directory("./data/label1", 1)
    
    features = []
    labels = []
    if feat0 is not None:
        features.append(feat0)
        labels.append(lbl0)
    if feat1 is not None:
        features.append(feat1)
        labels.append(lbl1)
        
    if not features:
        print("错误：没有加载到任何有效数据。")
        return

    data = np.vstack(features)
    label = np.vstack(labels).ravel() # 转换为一维数组以适配 SVM

    # 3. 随机划分
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, random_state=0, test_size=0.2
    )
    
    # 4. 训练与评估
    print(f"训练集规模: {x_train.shape}, 测试集规模: {x_test.shape}")
    clf = svm.SVC(kernel="rbf", gamma='scale')
    clf.fit(x_train, y_train)
    
    score = clf.score(x_test, y_test)
    print(f"准确率 (Accuracy): {score:.4f}")

if __name__ == "__main__":
    SVM_experiment()