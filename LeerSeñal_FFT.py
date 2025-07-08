import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import random

# Cargar el archivo .mat
mat = scipy.io.loadmat(r'D:\Mecatronica\Decimo semestre\SS\Paciente0332\0332_004_025_EEG.mat')
eeg = mat['val']  # eeg.shape debería ser (19, muestras)

print(f"Forma de la señal EEG: {eeg.shape}")

# Parámetros
fs = 256
minutos_a_eliminar = 5
muestras_a_eliminar = minutos_a_eliminar * 60 * fs

# Quitar offset
eeg_sin_offset = eeg - np.mean(eeg, axis=1, keepdims=True)

# Filtro Butterworth
lowcut = 0.1
highcut = 30.0
orden = 4
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = butter(orden, [low, high], btype='band')

# Filtrar señal
n_canales, n_muestras = eeg.shape
eeg_filtrada = np.zeros((n_canales, n_muestras - 2 * muestras_a_eliminar))
for i in range(n_canales):
    canal = eeg_sin_offset[i, muestras_a_eliminar : -muestras_a_eliminar]
    eeg_filtrada[i] = filtfilt(b, a, canal)

# FFT completa (global) del canal 1 (C4)
canal_fft = np.fft.fft(eeg_filtrada[1])
n = len(canal_fft)
frecuencias = np.fft.fftfreq(n, d=1/fs)
idx_positivos = frecuencias > 0
frecuencias = frecuencias[idx_positivos]
magnitud = np.abs(canal_fft[idx_positivos])
ventana = 20
paso = 1
magnitud_suavizada = np.convolve(magnitud, np.ones(ventana)/ventana, mode='valid')[::paso]
frecuencias_suavizadas = frecuencias[:len(np.convolve(magnitud, np.ones(ventana)/ventana, mode='valid'))][::paso]

# Graficar señal completa filtrada
plt.figure(figsize=(12, 4))
plt.plot(eeg_filtrada[1])
plt.title("Canal C4 - EEG filtrada entre 0.1 y 30 Hz (sin offset)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# Espectro completo
plt.figure(figsize=(12, 4))
plt.plot(frecuencias_suavizadas, magnitud_suavizada, color='orange')
plt.title("Espectro de Frecuencia Suavizado - Canal C4")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (promedio móvil)")
plt.grid(True)
plt.xlim([0, 30])
plt.show()

# Fragmentos aleatorios de 10s (5 min total)
duracion_total_s = eeg_filtrada.shape[1] / fs
max_inicio = int(duracion_total_s - 300)
inicio_aleatorio_s = random.randint(0, max_inicio)
inicio_muestra = inicio_aleatorio_s * fs
fin_muestra = inicio_muestra + 300 * fs
segmento_5min = eeg_filtrada[1, inicio_muestra:fin_muestra]
segmentos_10s = np.split(segmento_5min, 30)

# Graficar cada fragmento y su FFT (2 fragmentos por figura = 4 subplots)
for i in range(0, len(segmentos_10s), 2):
    plt.figure(figsize=(14, 10))
    
    for j in range(2):
        if i + j < len(segmentos_10s):
            fragmento = segmentos_10s[i + j]
            tiempo = np.arange(len(fragmento)) / fs
            fft_fragmento = np.fft.fft(fragmento)
            frecs = np.fft.fftfreq(len(fragmento), d=1/fs)
            idx_f = frecs > 0
            frecs = frecs[idx_f]
            magnitud_f = np.abs(fft_fragmento[idx_f])
            
            # Señal en el tiempo
            plt.subplot(4, 1, j * 2 + 1)
            plt.plot(tiempo, fragmento)
            plt.title(f'Segmento {i + j + 1} - Señal en el tiempo')
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud")
            plt.grid(True)
            
            # Espectro de frecuencia
            plt.subplot(4, 1, j * 2 + 2)
            plt.plot(frecs, magnitud_f)
            plt.title(f'Segmento {i + j + 1} - FFT')
            plt.xlabel("Frecuencia (Hz)")
            plt.ylabel("Magnitud")
            plt.grid(True)
            plt.xlim([0, 30])
    
    plt.tight_layout()
    plt.suptitle("Segmentos de 10s del canal C4 con su Transformada de Fourier", fontsize=14, y=1.02)
    plt.show()
