import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, welch
import pywt

plt.close("all")

# Sección 1: Señales ECG (1 derivación)

# Cargar datos ECG
ecgData = loadmat("D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_ECG.mat")
ecgSignals = np.array(ecgData["val"])
ecgSenal = ecgSignals[1,:]  # Usamos la segunda derivación

# Segmentación (30 segundos)
fs_ecg = 500
segment_duration = 30 * fs_ecg
ecg_segments = [ecgSenal[i*segment_duration:(i+1)*segment_duration] for i in range(len(ecgSenal)//segment_duration)]
ecgSignal = ecg_segments[3]  # Tomamos el cuarto segmento

# Sección 2: Filtrado ECG (Butterworth 0.1-100 Hz)
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

filtered_ecg = apply_bandpass_filter(ecgSignal, 0.1, 100, fs_ecg)

# Sección 3: Filtrado de ondas (Wavelet)
coeffs = pywt.wavedec(filtered_ecg, 'db4', level=3)
a3 = pywt.waverec([coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]], 'db4')[:len(filtered_ecg)]
d1 = pywt.waverec([np.zeros_like(coeffs[0])] + [coeffs[1]] + [np.zeros_like(d) for d in coeffs[2:]], 'db4')[:len(filtered_ecg)]
d2 = pywt.waverec([np.zeros_like(coeffs[0])] + [np.zeros_like(coeffs[1]), coeffs[2]] + [np.zeros_like(d) for d in coeffs[3:]], 'db4')[:len(filtered_ecg)]
wavelet_filtered = a3 + d1 + d2

# Sección 4: Algoritmo Pan-Tompkins
# Derivada
ecg_derivative = np.diff(wavelet_filtered)
# Cuadrado
ecg_squared = ecg_derivative ** 2
# Integración
window_size = int(0.12 * fs_ecg)
integration_window = np.ones(window_size) / window_size
ecg_integrated = np.convolve(ecg_squared, integration_window, mode='same')
# Detección de picos
peaks, _ = find_peaks(ecg_integrated, height=0.6*np.max(ecg_integrated), distance=int(0.2*fs_ecg))


# Sección 5: Señales EEG (20 derivaciones)
# Cargar datos EEG
eegData = loadmat("D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_EEG.mat")
eegSignals = np.array(eegData["val"])

# Nombres de canales
canales_eeg = ["C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", 
               "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6"]

# Segmentación (30 segundos)
fs_eeg = 500
eeg_segments = [eegSignals[:, i*segment_duration:(i+1)*segment_duration] for i in range(eegSignals.shape[1]//segment_duration)]
eegSignal_segment = eeg_segments[3][1,:]  # Tomamos el cuarto segmento, canal C4

# Sección 6: Filtrado EEG (Butterworth 0.05-30 Hz)

filtered_eeg = apply_bandpass_filter(eegSignal_segment, 0.05, 30, fs_eeg)


# Sección 7: Filtro Notch (50 Hz)
def notch_filter(data, fs, freq_to_remove=50, Q=30):
    nyq = 0.5 * fs
    freq = freq_to_remove / nyq
    b, a = butter(2, [freq-1/Q, freq+1/Q], btype='bandstop')
    y = filtfilt(b, a, data)
    return y

notch_filtered_eeg = notch_filter(filtered_eeg, fs_eeg)

# Sección 8: Segmentación por canales
# Ya realizado al cargar los datos

# Sección 9: Método de Welch

def compute_psd(signal, fs, nperseg=1024):
    f, Pxx = welch(signal, fs, nperseg=nperseg)
    return f, Pxx

# Aplicar a EEG
f_eeg, psd_eeg = compute_psd(notch_filtered_eeg, fs_eeg)

# Aplicar a ECG
f_ecg, psd_ecg = compute_psd(wavelet_filtered, fs_ecg)


# Sección 10: Exportación y compactación

plt.figure(figsize=(15, 10))

# ECG
plt.subplot(3, 2, 1)
plt.plot(ecgSignal)
plt.title("Señal ECG original")

plt.subplot(3, 2, 2)
plt.plot(filtered_ecg)
plt.title("ECG filtrado (0.1-100 Hz)")

plt.subplot(3, 2, 3)
plt.plot(wavelet_filtered)
plt.title("ECG filtrado con Wavelet")

plt.subplot(3, 2, 4)
plt.plot(ecg_integrated)
plt.plot(peaks, ecg_integrated[peaks], 'ro')
plt.title("Detección de picos R (Pan-Tompkins)")

# EEG
plt.subplot(3, 2, 5)
plt.plot(eegSignal_segment)
plt.title("Señal EEG original (C4)")

plt.subplot(3, 2, 6)
plt.plot(notch_filtered_eeg)
plt.title("EEG filtrado (0.05-30 Hz + Notch)")

plt.tight_layout()

# PSD
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(f_ecg, psd_ecg)
plt.title("Densidad espectral de potencia (ECG)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (V²/Hz)")

plt.subplot(1, 2, 2)
plt.semilogy(f_eeg, psd_eeg)
plt.title("Densidad espectral de potencia (EEG)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (V²/Hz)")

plt.tight_layout()
plt.show()

# Visualización de resultados - PSD combinada
plt.figure(figsize=(10, 6))

# Graficar PSD de ECG
plt.semilogy(f_ecg, psd_ecg, label='ECG', color='blue')

# Graficar PSD de EEG
plt.semilogy(f_eeg, psd_eeg, label='EEG', color='green')

plt.title("Densidad espectral de potencia (ECG vs EEG)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (V²/Hz)")
plt.legend()
plt.grid(True)

# Ajustar los límites del eje x para mejor visualización
plt.xlim([0, max(np.max(f_ecg), np.max(f_eeg))])

plt.tight_layout()
plt.show()