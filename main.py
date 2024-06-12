import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
# import scipy.fft as scipyfft
import timeit

import data.single_frequency_signal as sfs
import data.two_frequency_signal as tfs
import data.smooth_signal as sms

import fft as fft

def getTimeForFftAndDftForCompare(data):
    dft_time = timeit.timeit(lambda: fft.FFT(data), number=50)
    print('\nDFT time:', dft_time, 'secs.')
    fft_time = timeit.timeit(lambda: fft.FFT(data), number=50)
    print('FFT time:', fft_time, 'secs.')
    print('FFT', 'faster than' if (dft_time > fft_time) else 'slower than', 'DFT\n')

def removeTrend(data):
    plt.plot(data, label='original')
    # find large, rapid drops in amplitdue
    jmps = np.where(np.diff(data) < -0.5)[0]
    for j in jmps:
        data[j+1:] += data[j] - data[j+1]    
    # plt.plot(data, label='unrolled')
    # detrend with a low-pass
    order = 16
    data -= sps.filtfilt([1] * order, [order], data) 
    plt.plot(data, label='detrended')
    plt.legend(loc='best')
    plt.show()
    return data

def FFT(data):
    # fft_result = fft.FFT(data)
    # plt.plot(fft_result, label='fft-transform')

    fs = 1000 # ???
    fft_result = fft.FFT(data)
    fft_freq = np.fft.fftfreq(len(data), 1/fs)
    # Определение индексов положительных частот
    positive_freqs = fft_freq > 0
    # Получение магнитуды спектра и соответствующих частот
    magnitude = np.abs(fft_result[positive_freqs])
    frequencies = fft_freq[positive_freqs]
    # Нахождение верхней и нижней частоты сигнала
    lower_freq = frequencies[np.argmax(magnitude > 0)]
    upper_freq = frequencies[np.argmax(magnitude[::-1] > 0)]
    print(f"Нижняя частота сигнала: {lower_freq} Гц")
    print(f"Верхняя частота сигнала: {upper_freq} Гц")
    plt.plot(frequencies, magnitude)
    plt.title('Спектр сигнала')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.legend(loc='best')

    plt.show()

def main(data):
    data = np.asarray(data, dtype=float)
    data = removeTrend(data)
    print('Mathematical expectation:', np.mean(data)) # Математическое ожидание

    # # #
    # index = 0
    # p = 10
    # slicer = p - 1
    # result = np.asarray([], dtype=float)
    # while(slicer <= data.size):
    #     print(index, slicer)
    #     result = np.append(result, fft.FFT(data[index:slicer]), axis=None)
    #     slicer += p
    #     index += p
    # print(result.size)
    # plt.plot(result, label='fft-transform')
    # plt.legend(loc='best')
    # plt.show()
    # # #

    FFT(data)
    getTimeForFftAndDftForCompare(data)

# Одночастотный сигнал
single_frequency_data = sfs.getData()
plt.title('x=sin(wy)')
main(single_frequency_data)

# Двухчастотный сигнал
two_frequency_data = tfs.getData()
plt.title('x=αsin(w₁*y) + βsin(w₂*y)')
main(two_frequency_data)

# Отображение гладкой функции
smooth_displaying_data = sms.getData()
plt.title('xₙ₊₁=4(1-xₙ)xₙ')
main(smooth_displaying_data)
