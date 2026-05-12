import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import stats, signal
from scipy import interpolate
import math
from scipy import signal as spsignal

import time
# raw_data = pd.read_csv('data.csv').values

def Get_EMG_Feature(featureName, data, window=100, step=20, deadzone=1e-5):
    if featureName == "feat1":
        f1 = feature_mav1(data, window, step)
        f2 = feature_mavslp(data, window, step)
        f3 = feature_zc(data, window, step, deadzone)
        # f4 = feature_ssc(data, window, step, deadzone)
        f5 = feature_wl(data, window, step)
        return np.hstack((f1, f2, f3, f5))





def window_trapezoidal(size, slope):
    """
    Return trapezoidal window of length size, with each slope occupying slope*100% of window
    :param size: int - window length
    :param slope: float - trapezoid parameter, each slope occupies slope*100% of window
    :return: numpy.ndarray - trapezoidal window
    """
    arr = np.zeros(
        size,
    )
    if slope > 0.5:
        slope = 0.5
    if slope == 0:
        return np.full(size, 1)
    else:
        return np.array(
            [
                1
                if ((slope * size <= i) & (i <= (1 - slope) * size))
                else (1 / slope * i / size)
                if (i < slope * size)
                else (1 / slope * (size - i) / size)
                for i in range(1, size + 1)
            ]
        )


def moving_window_stride(array, window, step):
    """
    Returns view of strided array for moving window calculation with given window size and step
    :param array: numpy.ndarray - input array
    :param window: int - window size
    :param step: int - step lenght
    :return: strided: numpy.ndarray - view of strided array, index: numpy.ndarray - array of indexes
    """
    data_size = array.shape
    stride = array.strides
    win_count = math.floor((data_size[0] - window + step) / step)
    strided = as_strided(
        array,
        shape=(win_count, window, data_size[1]),
        strides=(step * stride[0], stride[0], stride[1]),
    )
    index = np.arange(window - 1, window + (win_count - 1) * step, step)
    return strided, index


def feature_iav(series, window, step):
    """Integral Absolute Value"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.sum(np.abs(windows_strided), axis=1)


def feature_aac(series, window, step):
    """Average Amplitude Change"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.divide(np.sum(np.abs(np.diff(windows_strided, axis=1)), axis=1), window)


def feature_apen(series, window, step, m, r):
    """Approximate Entropy
    AnEn feature is using PyEEG library v0.4.0 as it is, licensed with GNU GPL v3
    http://pyeeg.org"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.apply_along_axis(
        lambda win: pyeeg.ap_entropy(win, m, r), axis=1, arr=windows_strided
    )


def feature_ar(series, window, step, order) -> pd.DataFrame:
    """Auto-Regressive Coefficients 未改"""

    windows_strided, indexes = moving_window_stride(series, window, step)

    column_names = [str(i) for i in range(0, order)]
    win_coefs = pd.DataFrame(
        index=range(indexes[0]), columns=column_names, dtype=np.float64
    )

    for widx in range(len(windows_strided)):
        stride = windows_strided[widx].strides[0]
        stride_count = len(windows_strided[widx]) - order
        x = as_strided(
            windows_strided[widx], shape=[stride_count, order], strides=(stride, stride)
        )
        y = windows_strided[widx][order:]

        a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

        win_coefs.loc[series.index[indexes[widx]], :] = a
    return win_coefs


def feature_cc(series, window, step, order):
    """Cepstral Coefficients未改"""
    win_coefs = feature_ar(series, window, step, order)
    coefs = win_coefs
    coefs[:, 0] = -coefs[:, 0]
    for r in range(0, coefs.shape[0]):
        for p in range(1, order):
            coefs[r, p] = -coefs[r, p] - np.sum(
                [1 - (l / (p + 1)) for l in range(1, p + 1)]
                * np.full(p, coefs[r, p] * coefs[r, p - 1])
            )
    win_coefs.loc[:, :] = coefs
    return win_coefs


def feature_dasdv(series, window, step):
    """Difference Absolute Standard Deviation Value"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.sqrt(np.mean(np.square(np.diff(windows_strided, axis=1)), axis=1))


def feature_kurt(series, window, step):
    """Kurtosis"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return stats.kurtosis(windows_strided, axis=1)


def feature_log(series, window, step):
    """Log Detector"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.exp(np.mean(np.log(np.abs(windows_strided)), axis=1))


def feature_mav1(series, window, step):
    """Modified Mean Absolute Value Type 1"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    win_weight = [
        1 if ((0.25 * window <= i) & (i <= 0.75 * window)) else 0.5
        for i in range(1, window + 1)
    ]
    win_weight = np.array(win_weight)
    win_weight = np.tile(np.expand_dims(win_weight, 1), (1, windows_strided.shape[2]))
    return np.mean(np.abs(windows_strided) * win_weight, axis=1)


def feature_mav2(series, window, step):
    """Modified Mean Absolute Value Type 2"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    win_weight = window_trapezoidal(window, 0.25)
    win_weight = np.tile(np.expand_dims(win_weight, 1), (1, windows_strided.shape[2]))
    return np.mean(np.abs(windows_strided) * win_weight, axis=1)


def feature_mav(series, window, step):
    """Mean Absolute Value"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.mean(np.abs(windows_strided), axis=1)


def feature_mavslp(series, window, step):
    """Mean Absolute Value Slope"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    data = np.mean(
        np.abs(
            windows_strided[
                :, (int)(windows_strided.shape[1] / 2) : windows_strided.shape[1], :
            ]
        ),
        axis=1,
    ) - np.mean(
        np.abs(windows_strided[:, 0 : (int)(windows_strided.shape[1] / 2), :]), axis=1
    )

    return data


def feature_mhw(series, window, step):
    """Multiple Hamming Windows"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    win_weight = np.hamming(window)
    win_weight = np.tile(np.expand_dims(win_weight, 1), (1, windows_strided.shape[2]))
    return np.sum(np.square(windows_strided * win_weight), axis=1)


def feature_mtw(series, window, step, windowslope):
    """Multiple Trapezoidal Windows"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.sum(
        np.square(windows_strided) * window_trapezoidal(window, windowslope), axis=1
    )


def feature_myop(series, window, step, threshold):
    """Myopulse Percentage Rate"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.sum(windows_strided > threshold, axis=1) / window


def feature_rms(series, window, step):
    """Root Mean Square"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.sqrt(np.mean(np.square(windows_strided), axis=1))


def feature_sampleen(series, window, step, m, r):
    """Sample Entropy
    SampEn feature is using PyEEG library v 0.02_r2 as it is, licensed with GNU GPL v3
    http://pyeeg.sourceforge.net/"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.apply_along_axis(
        lambda win: pyeeg.samp_entropy(win, m, r), axis=1, arr=windows_strided
    )


def feature_skew(series, window, step):
    """Skewness"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return stats.skew(windows_strided, axis=1)


def feature_ssc(series, window, step, threshold):
    """Slope Sign Change"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.apply_along_axis(
        lambda x: np.sum((np.diff(x[:-1]) * np.diff(x[1:])) <= -threshold),
        axis=1,
        arr=windows_strided,
    )


def feature_ssi(series, window, step):
    """Simple Square Integral"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.sum(np.square(windows_strided), axis=1)


def feature_tm(series, window, step, order):
    """Absolute Temporal Moment"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.abs(np.mean(np.power(windows_strided, order), axis=1))


def feature_var(series, window, step):
    """Variance"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.var(windows_strided, axis=1)


def feature_v(series, window, step, v):
    """V-Order"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.power(np.abs(np.mean(np.power(windows_strided, v), axis=1)), 1.0 / v)


def feature_wamp(series, window, step, threshold):
    """Willison Amplitude"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.sum(np.diff(windows_strided, axis=1) >= threshold, axis=1)


def feature_wl(series, window, step):
    """Waveform Length"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.sum(np.diff(windows_strided, axis=1), axis=1)


def feature_zc(series, window, step, threshold):
    """Zero Crossing"""
    windows_strided, indexes = moving_window_stride(series, window, step)

    zc = np.sum(
        np.where(
            np.where(windows_strided < threshold, windows_strided, 0) > -threshold, 1, 0
        ),
        axis=1,
    )

    return zc


def feature_mnf(series, window, step):
    """Mean Frequency"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    windows_strided = np.squeeze(windows_strided)
    freq, power = signal.periodogram(windows_strided, 2000, axis=0)
    freq = np.expand_dims(freq, axis=1)
    data = np.sum(power * freq, axis=0) / np.sum(power, axis=0)
    data = np.expand_dims(data, 1)
    return


def feature_mdf(series, window, step):
    """Median Frequency"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    ttp_half = np.sum(power, axis=1) / 2

    power = np.squeeze(power)
    mdf = np.zeros(power.shape[1])
    for w in range(power.shape[1]):
        for s in range(1, power.shape[0]):
            if np.sum(power[0:s, w]) > ttp_half[0, w]:
                mdf[w] = freq[s - 1]
                break
    mdf = np.expand_dims(mdf, axis=0)
    return mdf


def feature_pkf(series, window, step):
    """Peak Frequency"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    return freq[np.argmax(power, axis=1)]


def feature_mnp(series, window, step):
    """Mean Power"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    return np.mean(power, axis=1)


def feature_ttp(series, window, step):
    """Total Power"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    return np.sum(power, axis=1)


def feature_sm(series, window, step, order):
    """Spectral Moment"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    freq = np.expand_dims(freq, axis=1)

    return np.sum(power * np.power(freq, order), axis=1)


def feature_fr(series, window, step, flb, fhb):
    """Frequency Ratio"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    lb = np.sum(power[:, (flb[0] < freq) & (freq < flb[1])], axis=1)
    hb = np.sum(power[:, (fhb[0] < freq) & (freq < fhb[1])], axis=1)
    return lb / hb


def feature_vcf(series, window, step):
    """Variance of Central Frequency"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    power = np.squeeze(power)
    freq = np.expand_dims(freq, axis=1)

    def sm(order):
        return np.sum(power * np.power(freq, order), axis=0)

    data = sm(2) / sm(0) - np.square(sm(1) / sm(0))
    data = np.expand_dims(data, axis=0)
    return


def feature_psr(series, window, step):
    """Power Spectrum Ratio"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    PKF_id = np.argmax(power, axis=1)
    lb = np.where(PKF_id - 20 < 0, 0, PKF_id - 20)
    hb = np.where(PKF_id + 20 > window, window, PKF_id + 20)
    data = np.zeros([1, 9])
    power = np.squeeze(power)
    for i in range(lb.shape[1]):
        data[0, i] = sum(power[lb[0, i] : hb[0, i], i]) / np.sum(power[:, i])
    return data


def feature_snr(series, window, step, powerband, noiseband1, noiseband2):
    """Signal-to-Noise Ratio"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    snr = np.apply_along_axis(
        lambda p: np.sum(p[(freq > powerband[0]) & (freq < powerband[1])])
        / (
            np.sum(
                p[
                    (freq > noiseband1[0])
                    & (freq < noiseband1[1])
                    & (freq > noiseband2[0])
                    & (freq < noiseband2[1])
                ]
            )
            * np.max(freq)
        ),
        axis=1,
        arr=power,
    )
    return snr


def feature_dpr(series, window, step, band, n):
    """Maximum-to-minimum Drop in Power Density Ratio"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)

    dpr = pd.Series()
    for pidx in range(len(power)):
        power_b = power[pidx][(freq > band[0]) & (freq < band[1])]
        stride = power_b.strides[0]
        stride_count = len(power_b) - n + 1
        p_strided = as_strided(
            power_b, shape=[stride_count, n], strides=(stride, stride)
        )
        means = np.mean(p_strided, axis=1)
        dpr.at[series.index[indexes[pidx]]] = np.max(means) / np.min(means)

    return dpr.values


def feature_ohm(series, window, step):
    """Power Spectrum Deformation"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    power = np.squeeze(power)
    freq = np.expand_dims(freq, axis=1)

    def sm(order):
        return np.sum(power * np.power(freq, order), axis=0)

    data = np.sqrt(sm(2) / sm(0)) / (sm(1) / sm(0))
    data = np.expand_dims(data, axis=0)
    return data


def feature_max(series, window, step, order, cutoff):
    """Maximum Amplitude"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    fs = 2000
    b, a = signal.butter(
        order, cutoff / (0.5 * fs), btype="lowpass", analog=False, output="ba"
    )
    return np.max(signal.lfilter(b, a, np.abs(windows_strided), axis=1), axis=1)


def feature_smr(series, window, step, n):
    """Signal-to-Motion Artifact Ratio"""
    # TODO: Verification Needed
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)

    freq_over35 = freq > 35
    freq_over35_idx = np.argmax(freq_over35)

    smr = pd.Series()
    for pidx in range(len(power)):
        power_b = power[pidx][freq_over35]
        stride = power_b.strides[0]
        stride_count = len(power_b) - n + 1
        p_strided = as_strided(
            power_b, shape=[stride_count, n], strides=(stride, stride)
        )
        mean = np.mean(p_strided, axis=1)
        max = np.max(mean)
        max_idx = np.argmax(mean) + int(np.floor(n / 2.0)) + freq_over35_idx
        a = max / freq[max_idx]

        smr.at[series.index[indexes[pidx]]] = np.sum(power[pidx][freq < 600]) / np.sum(
            power[pidx][power[pidx] > (freq * a)]
        )

    return smr.values


def box_counting_dimension(sig, y_box_size_multiplier, subsampling):
    # Box-Counting Example:
    # https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0#file-fractal-dimension-py-L25
    n = 2 ** np.floor(np.log(len(sig)) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)

    box_count = []
    for box_size in sizes:
        x_box_size = box_size
        y_box_size = box_size * y_box_size_multiplier

        sig_minimum = np.min(sig)

        box_occupation = np.zeros(
            [
                int(len(sig) / x_box_size) + 1,
                int((np.max(sig) - sig_minimum) / y_box_size) + 1,
            ]
        )

        interp_func = interpolate.interp1d(
            np.arange(0, len(sig), 1), sig.reshape(1, len(sig))[0]
        )
        x_interp = np.arange(0, len(sig) - 1 + 1 / subsampling, 1 / subsampling)
        sig_interp = interp_func(x_interp)

        for i in range(len(sig_interp)):
            x_box_id = int(x_interp[i] / x_box_size)
            y_box_id = int((sig_interp[i] - sig_minimum) / y_box_size)
            box_occupation[x_box_id, y_box_id] = 1

        box_count.append(np.sum(box_occupation))

    coefs = np.polyfit(np.log(1 / sizes), np.log(box_count), 1)
    return coefs[0]


def feature_bc(series, window, step, y_box_size_multiplier, subsampling):
    """Box-Counting Dimension"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return (
        np.apply_along_axis(
            lambda sig: box_counting_dimension(sig, y_box_size_multiplier, subsampling),
            axis=1,
            arr=windows_strided,
        ),
    )


def feature_psdfd(series, window, step, power_box_size_multiplier, subsampling):
    """Power Spectral Density Fractal Dimension"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    freq, power = signal.periodogram(windows_strided, 2000, axis=1)
    return np.apply_along_axis(
        lambda sig: box_counting_dimension(sig, power_box_size_multiplier, subsampling),
        axis=1,
        arr=power,
    )


def force_feature_mean(series, window, step):
    """Mean value"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.mean(windows_strided, axis=1)


def force_feature_median(series, window, step):
    """Median value"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return np.median(windows_strided, axis=1)


def force_feature_last(series, window, step):
    """Last value of the window - resampling"""
    windows_strided, indexes = moving_window_stride(series, window, step)
    return windows_strided[::, -1]
