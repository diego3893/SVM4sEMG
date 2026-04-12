import numpy as np
from scipy import signal
from features import Get_EMG_Feature
from sklearn import svm
from sklearn.model_selection import train_test_split


def load_data():
    data = np.loadtxt("data.txt")
    return data


def data_filter(data):
    # 带通滤波
    fs = 2000
    low_fs = 20
    high_fs = 200
    b, a = signal.butter(
        4, [2 * low_fs / fs, 2 * high_fs / fs], "bandpass"
    )  # 4阶巴特沃兹滤波器
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    # 陷波滤波器
    low_fs1 = 48
    high_fs1 = 52
    b1, a1 = signal.butter(
        4, [2 * low_fs1 / fs, 2 * high_fs1 / fs], "bandstop"
    )  # 4阶巴特沃兹滤波器
    filtered_data = signal.filtfilt(b1, a1, filtered_data, axis=0)
    return filtered_data


def data_process(filename, label):
    data = np.load(filename)
    filtered_data = data_filter(data)
    win_len = 100
    win_move = 20
    # 滑窗处理

    num_all = int((filtered_data.shape[0] - win_len) / win_move) + 1
    data_ = np.zeros((num_all, win_len, filtered_data.shape[1]))
    for i in range(num_all):
        data_[i, :, :] = filtered_data[i * win_move : i * win_move + win_len]
    swg = 1
    data_feature = []
    for i in range(num_all):
        data_feature.append(Get_EMG_Feature("feat1", data_[i, :, :]))
    data_feature = np.array(data_feature).squeeze()
    data_label = np.ones((num_all, 1)) * label
    return data_feature, data_label


def SVM_experiment():
    # 可能要改？data_process里面的feat1需要改吗？
    data, label = data_process("../data/data1.npy", 1) # 加载数据
    # 随机划分
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, random_state=0, test_size=0.2
    )
    # 加载模型
    clf = svm.SVC(kernel="rbf")
    # 训练
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy:", np.mean(y_pred == y_test))


if __name__ == "__main__":
    SVM_experiment()
