import numpy as np
import scipy.io as sio
import scipy.signal as signal
import pywt
import matplotlib.pyplot as plt

# Cargar archivo .mat con señal ECG
ruta = r"D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_ECG.mat"
data = sio.loadmat(ruta)
ecgSenal = data['val'][0]  

# Definir duración del segmento
segment_duration = 6000  # 30 segundos * 200 Hz

# Función para dividir una señal en segmentos de duración específica
def split_signal_into_segments(signal, segment_size):
    num_segments = len(signal) // segment_size  # Número de segmentos completos
    segments = [signal[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
    return np.array(segments)  # Convertir a arreglo numpy

# Cortar señal
ecg_segments = split_signal_into_segments(ecgSenal, segment_duration)
ecg_signal = ecg_segments[3]

fs = 200  # 200 Hz
t = np.arange(len(ecg_signal)) / fs  # Vector de tiempo

# DFT vs FFT
N = len(ecg_signal)
dft = np.array([sum(ecg_signal * np.exp(-2j * np.pi * k * np.arange(N) / N)) for k in range(N)])
fft = np.fft.fft(ecg_signal)
freqs = np.fft.fftfreq(N, 1/fs)

plt.figure(figsize=(10, 4))
plt.plot(freqs[:N//2], np.abs(dft[:N//2]), label="DFT (Manual)", linestyle='dashed')
plt.plot(freqs[:N//2], np.abs(fft[:N//2]), label="FFT (Optimizada)", alpha=0.7)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.title("Comparación DFT vs FFT")
plt.legend()
plt.show()

# Transformada Wavelet Discreta (DWT) con Daubechies 3
coeffs = pywt.wavedec(ecg_signal, 'db3', level=3)

plt.figure(figsize=(8, 6))
for i, coef in enumerate(coeffs):
    plt.subplot(len(coeffs), 1, i+1)
    plt.plot(coef)
    plt.title(f'Nivel {i} - Coeficientes')

plt.tight_layout()
plt.show()

# Espectrograma (STFT - Transformada de Fourier de Ventana Corta)
frequencies, times, Sxx = signal.spectrogram(ecg_signal, fs=fs, nperseg=100)

plt.figure(figsize=(8, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.title('Espectrograma del ECG')
plt.colorbar(label='Amplitud (dB)')
plt.show()
