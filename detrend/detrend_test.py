import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps

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