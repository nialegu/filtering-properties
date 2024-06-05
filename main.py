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
    dft_time = timeit.timeit(lambda: fft.FFT(data), number=1)
    print('\nDFT time:', dft_time, 'secs.')
    fft_time = timeit.timeit(lambda: fft.FFT(data), number=1)
    print('FFT time:', fft_time, 'secs.')
    print('FFT', 'faster than' if (dft_time > fft_time) else 'slower than', 'DFT\n')

def removeTrend(data):
    plt.plot(data, label='original')
    # detect and remove jumps
    jmps = np.where(np.diff(data) < -0.5)[0]  # find large, rapid drops in amplitdue
    for j in jmps:
        data[j+1:] += data[j] - data[j+1]    
    plt.plot(data, label='unrolled')
    # detrend with a low-pass
    order = 16
    data -= sps.filtfilt([1] * order, [order], data)  # this is a very simple moving average filter
    plt.plot(data, label='detrended')
    plt.legend(loc='best')
    return data

def FFT(data):
    result = fft.FFT(data)
    # result = scipyfft.fft(data)
    # print("Result:", result)
    plt.plot(result, label='fft-transform')
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
main(single_frequency_data)

# Двухчастотный сигнал
two_frequency_data = tfs.getData()
main(two_frequency_data)

# Отображение гладкой функции
smooth_displaying_data = sms.getData()
main(smooth_displaying_data)



# Ни фильтрация, ни простое определение тренда не принесут вам никакой пользы с этим сигналом. 
# Первая проблема заключается в том, что тренд носит несколько периодический характер. 
# Вторая проблема заключается в том, что периодичность не является стационарной. 
# Я считаю, что линейные методы не решат проблему.
# Я предлагаю вам сделать следующее:

# 1) удалить переходы ("развернуть пилообразный сигнал")
# 2) затем измените тренд сигнала с помощью фильтра нижних частот или чего-то еще
# Вот пример:

np.random.seed(123)

# create an example signal
x = []
ofs = 3.4
slope = 0.002
for t in np.linspace(0, 100, 1000):
    ofs += slope    
    x.append(np.sin(t*2) * 0.1 + ofs)
    if x[-1] > 4:
        ofs =3.2
        slope = np.random.rand() * 0.003 + 0.002
x = np.asarray(x)    
plt.plot(x, label='original')

# detect and remove jumps
jmps = np.where(np.diff(x) < -0.5)[0]  # find large, rapid drops in amplitdue
for j in jmps:
    x[j+1:] += x[j] - x[j+1]    
plt.plot(x, label='unrolled')

# detrend with a low-pass
order = 200
x -= sps.filtfilt([1] * order, [order], x)  # this is a very simple moving average filter
plt.plot(x, label='detrended')

plt.legend(loc='best')
# FFT(x)
plt.show()