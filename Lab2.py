# -----------------------------------------------------------
# Curso: Redes de Computadores
# Laboratorio 2: Modulación Analógica
# Integrantes: Angel Nieva
# Profesor: Carlos Gonzales
# Ayudante: Nicole Reyes
# github: https://github.com/Angel-Nieva/Lab2Redes.git
# -----------------------------------------------------------
from funciones.graficos import graficar
from funciones.modulacion import fourier_transform, am_modulation, am_demodulation, fm_modulation

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as waves
# -----------------------------------------------------------
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
graficar("Grafico 1.2: Señal original zoom", t, audio, "Tiempo [s]", "Amplitud")
plt.xlim(0, 0.01)
plt.ylim(-8000, 8000)
graficar("Grafico 1.3: Transformada de Fourier señal", signal_freq, np.abs(signal_tf), "Frecuencia [Hz]", "Amplitud")
plt.show()

modulation_freq = 10000
# ------------------------- Modulacion AM ----------------------------------#

resampling_time = np.linspace(0, tiempo, 10 * largo)
carrier_wave = np.cos(2 * np.pi*modulation_freq*resampling_time)

# Con k = 1

# Modulación AM:
am = am_modulation(audio, carrier_wave, t, resampling_time, k=1)

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
graficar("Grafico 3.2: Señal modulada zoom - AM", resampling_time, am, "Tiempo [s]", "Amplitud")
plt.xlim(0, 0.01)
plt.ylim(-8000, 8000)
graficar("Grafico 3.3: Transformada de Fourier - AM", am_freq, np.abs(am_tf), "Frecuencia [Hz]", "Amplitud")
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
graficar("Grafico 5.2: Transformada de Fourier demodulada - AM", demodulate_am_freq, np.abs(demodulate_am_tf),
         "Frecuencia [Hz]", "Amplitud")
plt.xlim(-4200, 4200)
plt.show()

# Con k = 2

# Modulación AM:
am = am_modulation(audio, carrier_wave, t, resampling_time, k=1.25)

# Transformada de Fourier señal AM:
am_tf, am_freq = fourier_transform(am, largo * 10, T * 0.1)

graficar("Grafico 6.1: Señal modulada - AM", resampling_time, am, "Tiempo [s]", "Amplitud")
graficar("Grafico 6.2: Señal  zoom - AM", resampling_time, am, "Tiempo [s]", "Amplitud")
plt.xlim(0, 0.01)
plt.ylim(-8000, 8000)
graficar("Grafico 6.3: Transformada de Fourier - AM", am_freq, np.abs(am_tf), "Frecuencia [Hz]", "Amplitud")
plt.show()

# Se demodula la señal AM
demodulated_am = am_demodulation(am, carrier_wave, 10*faudio, 8000)

# Transformada de Fourier demodulacion AM:
demodulate_am_tf, demodulate_am_freq = fourier_transform(demodulated_am, largo * 10, T * 0.1)

graficar("Grafico 7.1: Señal demodulada - AM", resampling_time, demodulated_am, "Tiempo [s]", "Amplitud")
graficar("Grafico 7.2: Transformada de Fourier demodulada - AM", demodulate_am_freq, np.abs(demodulate_am_tf),
         "Frecuencia [Hz]", "Amplitud")
plt.xlim(-4200, 4200)
plt.show()
# ------------------------- Modulacion FM ----------------------------------#

tiempo_resample = np.linspace(0,int(tiempo),int(carrier_freq*tiempo))
port_resample = np.linspace(0,int(carrier_freq),int(carrier_freq*tiempo))


# Con k = 1
fm_10 = fm_modulation(audio, modulation_freq, t, resampling_time, 1)
fm_10_tf, fm_10_freq = fourier_transform(fm_10, largo * 10, T * 0.1)

graficar("Grafico 8.1: Señal modulada zoom - FM 10[kHz]", resampling_time, fm_10, "Tiempo [s]", "Amplitud")
plt.xlim(0, 0.01)
graficar("Grafico 8.2: Transformada de Fourier - FM", fm_10_freq, np.abs(fm_10_tf), "Frecuencia [Hz]", "Amplitud")
plt.show()

fm_20 = fm_modulation(audio, 20000, t, resampling_time, 1)
fm_20_tf, fm_20_freq = fourier_transform(fm_20, largo * 10, T * 0.1)

graficar("Grafico 9.1: Señal modulada zoom - FM 20[kHz]", resampling_time, fm_20, "Tiempo [s]", "Amplitud")
plt.xlim(0, 0.01)
graficar("Grafico 9.2: Transformada de Fourier - FM", fm_20_freq, np.abs(fm_20_tf), "Frecuencia [Hz]", "Amplitud")
plt.show()


dx = resampling_time[1]-resampling_time[0]
derivada = np.gradient(fm_20, dx)

fm_der_tf, fm_der_freq = fourier_transform(derivada, largo * 10, T * 0.1)
graficar("Grafico 10.1: Señal FM derivada - FM 20[kHz]", resampling_time, derivada, "Tiempo [s]", "Amplitud")
graficar("Grafico 10.2: Transformada de Fourier derivada - FM", fm_der_freq, np.abs(fm_der_tf), "Frecuencia [Hz]", "Amplitud")
plt.xlim(-100000, 100000)
plt.show()

