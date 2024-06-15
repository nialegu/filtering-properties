import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
# import scipy.fft as scipyfft
import timeit

import data.single_frequency_signal as sfs
import data.two_frequency_signal as tfs
import data.smooth_signal as sms
import data.two_dimensions_signal as tds

import fft as fft

def getTimeForFftAndDftForCompare(data):
    dft_time = timeit.timeit(lambda: fft.FFT(data), number=50)
    print('\nDFT time:', dft_time, 'secs.')
    fft_time = timeit.timeit(lambda: fft.FFT(data), number=50)
    print('FFT time:', fft_time, 'secs.')
    print('FFT', 'faster than' 
          if (dft_time > fft_time) 
          else 'slower than', 'DFT\n')

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
    # # Нахождение верхней и нижней частоты сигнала
    # lower_freq = frequencies[np.argmax(magnitude > 0)]
    # upper_freq = frequencies[np.argmax(magnitude[::-1] > 0)]
    # print(f"Нижняя частота сигнала: {lower_freq} Гц")
    # print(f"Верхняя частота сигнала: {upper_freq} Гц")
    plt.plot(frequencies, magnitude)
    plt.title('Спектр сигнала')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.legend(loc='best')

def signalDivider(data):
    index = 0
    p = 10
    slicer = p - 1
    result = np.asarray([], dtype=float)
    while(slicer <= data.size):
        print(index, slicer)
        result = np.append(result, fft.FFT(data[index:slicer]), axis=None)
        slicer += p
        index += p
    return result

def test(data):
    i = 0
    last_index = 0
    result = []
    for n in data:
        if (i != 0 and data[i-1]<0<n):
            result.append(data[last_index:i])
            if (round(np.mean(data[last_index:i])) == 0):
                print("?")
            last_index = i+1
        i += 1
    
    return result


def main(data):
    data = np.asarray(data, dtype=float)
    data = removeTrend(data)
    print('Mathematical expectation:', np.mean(data))
    data = test(data)
    # for arr in data:
    #     FFT(arr)
    FFT(data)
    getTimeForFftAndDftForCompare(data)
    plt.show()

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

# # # -------------------- # # #

def phase_portrait(data):
    y_n = data
    y_n_plus_1 = y_n[1:] + [0]
    plt.plot(y_n[:-1], y_n_plus_1, 'bo', label='Точки (yₙ, yₙ₊₁)')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('yₙ')
    plt.ylabel('yₙ₊₁')
    plt.title('Фазовый портрет для дискретной системы')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def tds_main(data_1, data_2, data_3):
    phase_portrait(data_1)
    phase_portrait(data_2)
    phase_portrait(data_3)

    plt.plot(data_1, label='original')
    plt.legend(loc='best')
    plt.show()
    # data_1 = removeTrend(data_1)
    # data_2 = removeTrend(data_2)
    # data_3 = removeTrend(data_3)

    FFT(data_1)
    FFT(data_2)
    FFT(data_3)
    plt.show()  

tds_1 = tds.getFirstData()
tds_2 = tds.getSecondData()
tds_3 = tds.getThirdData()
plt.title('yₙ₊₁=f(ayₙ+byₙ₋₁)')
tds_main(tds_1, tds_2, tds_3)

#########################################

# # Задаем систему уравнений, например, x' = y, y' = -x
# def system(state, t):
#     x, y = state
#     return y, -x

# # Создаем сетку точек для фазового портрета
# y1, y2 = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))

# # Вычисляем производные в каждой точке сетки
# u, v = np.zeros(y1.shape), np.zeros(y2.shape)
# NI, NJ = y1.shape
# for i in range(NI):
#     for j in range(NJ):
#         x = y1[i, j]
#         y = y2[i, j]
#         state_derivatives = system([x, y], 0)
#         u[i,j] = state_derivatives[0]
#         v[i,j] = state_derivatives[1]

# # Нормализуем стрелки
# norm = np.sqrt(u**2 + v**2)
# u /= norm
# v /= norm

# # Рисуем фазовый портрет
# plt.quiver(y1, y2, u, v, norm, cmap=plt.cm.jet)
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Фазовый портрет системы')
# plt.show()