import numpy as np
import scipy.fft as scf
import scipy.fftpack as fourier
import scipy.signal as scs
from scipy import integrate

# Objetivo: Calcular la transformada de Fourier de una señal de audio.
# signal: señal de audio
# size: largo de la señal
# period: periodo de la señal
# Return:
#   tf: Transformada de Fourier de la señal
#   fs: arreglo de frecuencias de la transformada de Fourier
def fourier_transform(signal, size, period) -> object:
    # Transformada de fourier
    tf = scf.fft(signal)
    # Muestras de frecuencias
    fs = fourier.fftfreq(size, period)
    return tf, fs


# Objetivo: Calcular la modulación AM de una señal.
# signal: señal de audio
# carrier_wave: señal portadora
# ts: arreglo de tiempo de la señal de audio
# tc: arreglo de tiempo de la señal portadora
# k: índice de modulación
# Return:
#   signal_am: señal modulada AM
def am_modulation(signal, carrier_wave, ts, tc, k):
    # Interpolacion
    resampling_signal = np.interp(tc, ts, signal)
    # Señal modulada
    signal_am = k * resampling_signal * carrier_wave
    return signal_am

# Objetivo: Demodular una señal AM.
# signal: señal de audio
# carrier_wave: señal portadora
# sample_rate: frecuencia de muestreo señal de audio
# f: frecuencia de la portadora
# Return:
#   demodulate: señal AM demodulada
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

# Objetivo: Calcular la modulación FM de una señal.
# frec: frecuencia portadora
# ts: arreglo de tiempo de la señal de audio
# tc: arreglo de tiempo de la señal portadora
# k: índice de modulación
# Return:
#   signal_fm: señal modulada FM
def fm_modulation(signal, frec, ts, tc, k):
    xt = np.interp(tc, ts, signal)
    # Integral
    integral = integrate.cumtrapz(xt, tc, initial=0)
    w = frec * tc
    signal_fm = np.cos(2 * w * np.pi + k * integral * np.pi)
    return signal_fm
