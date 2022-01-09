import numpy as np
import scipy.fft as scf
import scipy.fftpack as fourier
import scipy.signal as scs
from scipy import integrate

def fourier_transform(signal, size, period):
    # Transformada de fourier
    tf = scf.fft(signal)
    # Muestras de frecuencias
    fs = fourier.fftfreq(size, period)
    return tf, fs


def am_modulation(signal, carrier_wave ,ts, tc, k):
    # Interpolacion
    resampling_signal = np.interp(tc, ts, signal)
    # Señal modulada
    signal_am = k * resampling_signal * carrier_wave
    return signal_am

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


def fm_modulation(signal, frec, ts, tc, k):
    xt = np.interp(tc, ts, signal)

    # Integral para obtener fi de t
    integral = integrate.cumtrapz(xt, tc, initial=0)
    #graficar("Grafico 6.1: Señal  - FM", tc, xt, "Tiempo [s]", "Amplitud")
    w = frec * tc
    signal_fm = np.cos(2 * w * np.pi + k * integral * np.pi)
    return signal_fm

def fm_demodulation(signal, frec, ts, tc, k):
    xt = np.interp(tc, ts, signal)

    # Integral para obtener fi de t
    integral = integrate.cumtrapz(xt, tc, initial=0)
    #graficar("Grafico 6.1: Señal  - FM", tc, xt, "Tiempo [s]", "Amplitud")
    w = frec * tc
    signal_fm = np.cos(2 * w * np.pi + k * integral * np.pi)
    return signal_fm