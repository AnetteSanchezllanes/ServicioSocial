import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, welch
import pywt

plt.close("all")

# =============================================
# Configuración de frecuencia de muestreo (200 Hz)
# =============================================
fs = 200  # Frecuencia de muestreo
nyq = 0.5 * fs  # Frecuencia de Nyquist (100 Hz)

# =============================================
# Funciones de filtrado mejoradas
# =============================================
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)  # Evita Wn = 0
    high = min(highcut / nyq, 0.999)  # Evita Wn = 1
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def notch_filter(data, fs, freq_to_remove=50, Q=30):
    nyq = 0.5 * fs
    freq = min(freq_to_remove / nyq, 0.999)  # Asegura Wn < 1
    bw = 1 / Q
    low = max(freq - bw/2, 0.001)
    high = min(freq + bw/2, 0.999)
    b, a = butter(2, [low, high], btype='bandstop')
    y = filtfilt(b, a, data)
    return y

# =============================================
# Función para calcular PSD optimizada
# =============================================
def compute_psd(signal, fs, nperseg=512, noverlap=256):
    """
    Calcula la PSD con parámetros optimizados para señales fisiológicas.
    """
    # Remover DC offset y normalizar
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    
    f, Pxx = welch(signal, fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    return f, Pxx

# =============================================
# Procesamiento de ECG
# =============================================
ecgData = loadmat("D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_ECG.mat")
ecgSignals = np.array(ecgData["val"])
ecgSenal = ecgSignals[1, :]  # Segunda derivación

# Segmentación (30 segundos)
segment_duration = 30 * fs
ecg_segments = [ecgSenal[i*segment_duration:(i+1)*segment_duration] 
                for i in range(len(ecgSenal)//segment_duration)]
ecgSignal = ecg_segments[3]  # Cuarto segmento

# Preprocesamiento ECG
ecgSignal = ecgSignal - np.mean(ecgSignal)  # Remover DC offset
ecgSignal = ecgSignal / np.max(np.abs(ecgSignal))  # Normalizar

# Filtrado ECG (0.1-60 Hz para señales cardíacas)
filtered_ecg = apply_bandpass_filter(ecgSignal, 0.5, 60, fs)

# Filtrado Wavelet
coeffs = pywt.wavedec(filtered_ecg, 'db4', level=3)
a3 = pywt.waverec([coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]], 'db4')[:len(filtered_ecg)]
d1 = pywt.waverec([np.zeros_like(coeffs[0])] + [coeffs[1]] + [np.zeros_like(d) for d in coeffs[2:]], 'db4')[:len(filtered_ecg)]
d2 = pywt.waverec([np.zeros_like(coeffs[0])] + [np.zeros_like(coeffs[1]), coeffs[2]] + [np.zeros_like(d) for d in coeffs[3:]], 'db4')[:len(filtered_ecg)]
wavelet_filtered = a3 + d1 + d2

# =============================================
# Procesamiento de EEG
# =============================================
eegData = loadmat("D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_EEG.mat")
eegSignals = np.array(eegData["val"])
eegSignal_segment = eegSignals[1, 3*segment_duration:4*segment_duration]  # Canal C4

# Preprocesamiento EEG
eegSignal_segment = eegSignal_segment - np.mean(eegSignal_segment)
eegSignal_segment = eegSignal_segment / np.max(np.abs(eegSignal_segment))

# Filtrado EEG (0.1-30 Hz)
filtered_eeg = apply_bandpass_filter(eegSignal_segment, 0.1, 30, fs)

# Filtro Notch (50 Hz)
notch_filtered_eeg = notch_filter(filtered_eeg, fs)

# =============================================
# Cálculo de PSD
# =============================================
f_ecg, psd_ecg = compute_psd(wavelet_filtered, fs)
f_eeg, psd_eeg = compute_psd(notch_filtered_eeg, fs)

# =============================================
# Visualización
# =============================================
plt.figure(figsize=(14, 10))

# Señal ECG temporal
plt.subplot(2, 2, 1)
plt.plot(np.arange(len(wavelet_filtered))/fs, wavelet_filtered)
plt.title("ECG filtrado (0.1-60 Hz + Wavelet)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (normalizada)")

# PSD ECG
plt.subplot(2, 2, 2)
plt.semilogy(f_ecg, psd_ecg)
plt.title("PSD de ECG (nperseg=512, noverlap=256)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia (V²/Hz)")
plt.grid(True)
plt.xlim([0, 60])  # Enfoque en frecuencias cardíacas

# Señal EEG temporal
plt.subplot(2, 2, 3)
plt.plot(np.arange(len(notch_filtered_eeg))/fs, notch_filtered_eeg)
plt.title("EEG filtrado (0.1-30 Hz + Notch 50 Hz)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (normalizada)")

# PSD EEG
plt.subplot(2, 2, 4)
plt.semilogy(f_eeg, psd_eeg)
plt.title("PSD de EEG (nperseg=512, noverlap=256)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia (V²/Hz)")
plt.grid(True)
plt.xlim([0, 30])  # Rango típico EEG

plt.tight_layout()
plt.show()
