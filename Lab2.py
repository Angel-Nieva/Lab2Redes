from graficos import graficar

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as waves
import scipy.fft as scf
import scipy.fftpack as fourier
import scipy.signal as scs
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.fftpack import fftshift


# -----------------------------------------------------------
# Curso: Redes de Computadores
# Laboratorio 2: Fourier en señales de audio
# Integrantes: Angel Nieva, Ándréz Araya
# Profesor: Carlos Gonzales
# Ayudante: Nicole Reyes
# github: https://github.com/Angel-Nieva/Lab2Redes.git
# -----------------------------------------------------------


def fourier_transform(signal, size, period):
    # Transformada de fourier
    tf = scf.fft(signal)
    # Muestras de frecuencias
    fs = fourier.fftfreq(size, period)
    return tf, fs


def am_modulation(signal, time, size, fc, k):
    # Tiempo de la señal
    time_signal = np.linspace(0, time, size)
    # Tiempo de la portadora
    tc = np.linspace(0, time, 10 * size)
    # Interpolacion
    resampling_signal = np.interp(tc, time_signal, signal)
    # Construccion portadora
    carrier_wave = np.cos(2 * np.pi*fc*tc)
    # Señal modulada
    signal_am = k * resampling_signal * carrier_wave
    return signal_am, carrier_wave, tc


def am_demodulation(signal, carrier, sample_rate, f):
    # Se multiplica un coseno a la señal modulada
    am_coseno = signal * carrier
    # Se aplica un filtro lowpass
    wn = f/sample_rate
    b, a = scs.butter(6, wn, "lowpass")
    demodulate = scs.filtfilt(b, a, am_coseno)
    # Se amplifica la señal para recuperar la amplitud original
    demodulate = 2*demodulate
    return demodulate


audio = './audio/handel.wav'

faudio, audio = waves.read(audio)
T = 1 / faudio
largo = len(audio)
tiempo = largo / faudio

t = np.linspace(0, tiempo, largo)

# Transformada de Fourier señal:
signal_tf, signal_freq = fourier_transform(audio, largo, T)

# Gráficos señal portadora y su TF:
graficar("Grafico 1.1: Señal original", t, audio, "Tiempo [s]", "Amplitud")
graficar("Grafico 1.2: Transformada de Fourier señal", signal_freq, np.abs(signal_tf), "Frecuencia [Hz]", "Amplitud")
plt.show()

# print(max(frec)) == 4100 Hz
# Teniendo en cuenta el teorema de muestreo, la frecuencia de la señal portadora debe ser al menos el doble
# que la frecuencia máxima de la señal, por lo que se ocuparan 10000 Hz para la portadora.

modulation_freq = 10000
# -----------------------------------------------------------#
# Estudiando la modulación con k = 1

# Modulación AM:
am, carrier_wave, resampling_time = am_modulation(audio, tiempo, largo, modulation_freq, k=1)

# Transformada de Fourier portadora:
carrier_tf, carrier_freq = fourier_transform(carrier_wave, largo * 10, T * 0.1)

# Gráficos señal portadora y su TF:
graficar("Grafico 2.1: Señal portadora", resampling_time, carrier_wave, "Tiempo [s]", "Amplitud")
plt.xlim(0.005, 0.008)
graficar("Grafico 2.2: Transformada de Fourier portadora", carrier_freq, np.abs(carrier_tf), "Frecuencia [Hz]",
         "Amplitud")
plt.grid(True)
plt.show()

# Transformada de Fourier señal AM:
am_tf, am_freq = fourier_transform(am, largo * 10, T * 0.1)

graficar("Grafico 3.1: Señal modulada - AM", resampling_time, am, "Tiempo [s]", "Amplitud")
graficar("Grafico 3.2: Transformada de Fourier - AM", am_freq, np.abs(am_tf), "Frecuencia [Hz]", "Amplitud")
plt.show()

# Se multiplica un coseno a la señal modulada
am_cos = am * carrier_wave

am_cos_tf, am_cos_freq = fourier_transform(am_cos, largo * 10, T * 0.1)
graficar("Grafico 4: Transformada de Fourier AM * cos", am_cos_freq, np.abs(am_cos_tf), "Frecuencia [Hz]", "Amplitud")
plt.show()

# Se demodula la señal AM
demodulated_am = am_demodulation(am, carrier_wave, 10*faudio, 8000)

# Transformada de Fourier demodulacion AM:
demodulate_am_tf, demodulate_am_freq = fourier_transform(demodulated_am, largo * 10, T * 0.1)

graficar("Grafico 5.1: Señal demodulada - AM", resampling_time, demodulated_am, "Tiempo [s]", "Amplitud")
graficar("Grafico 5.2: Transformada de Fourier demodulada - AM", demodulate_am_freq, np.abs(demodulate_am_tf), "Frecuencia [Hz]", "Amplitud")
plt.xlim(-4200, 4200)
plt.show()