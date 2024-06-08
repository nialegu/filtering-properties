import numpy as np
import matplotlib.pyplot as plt

def detrend(signal, t):
    # Вычисление тренда с помощью линейной регрессии
    A = np.vstack([t, np.ones(len(t))]).T
    m, c = np.linalg.lstsq(A, signal, rcond=None)[0]

    # Удаление тренда из сигнала
    detrended_signal = signal - (m*t + c)

    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(t, signal, label='Сигнал с трендом')
    plt.plot(t, m*t + c, 'r', label='Тренд')
    plt.legend()

    plt.subplot(212)
    plt.plot(t, detrended_signal, label='Сигнал без тренда')
    plt.legend()
    plt.show()

# Симуляция сигнала с трендом
t = np.linspace(0, 1, 100)
signal_with_trend = t + np.random.normal(size=t.size)
detrend(signal_with_trend, t)