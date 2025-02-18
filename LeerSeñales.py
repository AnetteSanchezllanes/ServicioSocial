import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks
import pywt

plt.close("all")

# Rutas de los archivos
direccion_ECG = "D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_ECG.mat"
direccion_EEG = "D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_EEG.mat"
direccion_TXT = open("D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332.txt")
direccion_Head_ECG = open("D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_ECG.hea")
direccion_Head_EEG = open("D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_EEG.hea")

# Lectura de archivos de texto
print("Archivo TXT")
print(direccion_TXT.read())
direccion_TXT.close()

print("Archivo Head ECG")
print(direccion_Head_ECG.read())
direccion_Head_ECG.close()

print("Archivo Head EEG")
print(direccion_Head_EEG.read())
direccion_Head_EEG.close()

# Cargar datos de los archivos .mat
ecgData = loadmat(direccion_ECG)
eegData = loadmat(direccion_EEG)

# Extracción de señales
#ecgSenal = np.array(list(ecgData.items())[0][1], dtype=np.float64)[0, :]
#eegSignal = np.array(list(eegData.items())[0][1], dtype=np.float64)[0, 26000:27000]

eegSignals = np.array(eegData["val"]) 
ecgSignals = np.array(ecgData["val"])

Ecg1 = ecgSignals[0,:]
ecgSenal = ecgSignals[1,:]

C3_eeg = eegSignals[0, :]  # Canal C3
eegSenal = eegSignals[1, :]  # Canal C4
O1_eeg = eegSignals[2, :]  # Canal O1
O2_eeg = eegSignals[3, :]  # Canal O2
Cz_eeg = eegSignals[4, :]  # Canal Cz
F3_eeg = eegSignals[5, :]  # Canal F3
F4_eeg = eegSignals[6, :]  # Canal F4
F7_eeg = eegSignals[7, :]  # Canal F7
F8_eeg = eegSignals[8, :]  # Canal F8
Fz_eeg = eegSignals[9, :]  # Canal Fz
Fp1_eeg = eegSignals[10, :]  # Canal Fp1
Fp2_eeg = eegSignals[11, :]  # Canal Fp2
Fpz_eeg = eegSignals[12, :]  # Canal Fpz
P3_eeg = eegSignals[13, :]  # Canal P3
P4_eeg = eegSignals[14, :]  # Canal P4
Pz_eeg = eegSignals[15, :]  # Canal Pz
T3_eeg = eegSignals[16, :]  # Canal T3
T4_eeg = eegSignals[17, :]  # Canal T4
T5_eeg = eegSignals[18, :]  # Canal T5
T6_eeg = eegSignals[19, :]  # Canal T6

# Definir duración del segmento (30 segundos en muestras)
segment_duration = 6000  # 30 segundos * 200 Hz

# Función para dividir una señal en segmentos de duración específica
def split_signal_into_segments(signal, segment_size):
    num_segments = len(signal) // segment_size  # Número de segmentos completos
    segments = [signal[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
    return np.array(segments)  # Convertir a arreglo numpy

# Cortar señales ECG y EEG
ecg_segments = split_signal_into_segments(ecgSenal, segment_duration)
eeg_segments = split_signal_into_segments(eegSenal, segment_duration)

ecgSignal = ecg_segments[3]
eegSignal = eeg_segments[3]

Prom_eegsignal = np.mean(eegSignal)
filtered_eeg = eegSignal - Prom_eegsignal

Prom_ecgsignal = np.mean(ecgSignal)
filtered_ecg = ecgSignal - Prom_ecgsignal

# Definición de filtro pasabanda Butterworth
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

# Parámetros de filtrado
fs_ecg = 500  # Frecuencia de muestreo para ECG (Hz)
fs_eeg = 500  # Frecuencia de muestreo para EEG (Hz)
lowcut_eeg, highcut_eeg = 0.1, 30
lowcut_ecg, highcut_ecg = 0.1, 60

# Filtrar señales
filtered_ecg = apply_bandpass_filter(ecgSignal, lowcut_ecg, highcut_ecg, fs_ecg)
filtered_eeg = apply_bandpass_filter(eegSignal, lowcut_eeg, highcut_eeg, fs_eeg)

# FFT para señales filtradas
N_ecg = len(filtered_ecg)
N_eeg = len(filtered_eeg)
fft_valores_ecg = np.fft.fft(filtered_ecg)
fft_frecuencias_ecg = np.fft.fftfreq(N_ecg, 1 / fs_ecg)
fft_magnitud_ecg = np.abs(fft_valores_ecg)

fft_valores_eeg = np.fft.fft(filtered_eeg)
fft_frecuencias_eeg = np.fft.fftfreq(N_eeg, 1 / fs_eeg)
fft_magnitud_eeg = np.abs(fft_valores_eeg)

coeffs = pywt.wavedec(filtered_ecg, 'db4', level=3)
c, l = coeffs[0], coeffs[1:]
filtered_ecgSignal_l = len(filtered_ecg)
t = np.arange(filtered_ecgSignal_l) / fs_ecg

# Reconstrucción de la última aproximación
a3 = pywt.waverec([coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]], 'db4')[:filtered_ecgSignal_l]

# Reconstrucción de los detalles
d1 = pywt.waverec([np.zeros_like(coeffs[0])] + [coeffs[1]] + [np.zeros_like(d) for d in coeffs[2:]], 'db4')[:filtered_ecgSignal_l]
d2 = pywt.waverec([np.zeros_like(coeffs[0])] + [np.zeros_like(coeffs[1]), coeffs[2]] + [np.zeros_like(d) for d in coeffs[3:]], 'db4')[:filtered_ecgSignal_l]

# Sumar la aproximación de nivel 5 y los detalles de nivel 1 a 4
suma_approx_detalles = a3 + d1 + d2 


plt.figure(figsize=(10, 6))
plt.plot(t, suma_approx_detalles, label='Aproximación + Detalles')
plt.title('Suma de la última aproximación y los detalles')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()

#for i, detail in enumerate(coeffs[1:], start=1):
#    plt.plot(detail)
#    plt.title(f'Detail {i}')
#    plt.xlabel('Samples')
#    plt.ylabel('Amplitude')

#plt.tight_layout()

# Algoritmo de Pan-Tompkins
# Paso 1: Derivada de la señal
ecg_derivative = np.diff(filtered_ecg)

# Paso 2: Cuadrado de la señal
ecq_squared = ecg_derivative ** 2

# Paso 3: Ventana de integración
window_size = int(0.12 * fs_ecg)  # Ventana de 120 ms
integration_window = np.ones(window_size) / window_size
ecq_integrated = np.convolve(ecq_squared, integration_window, mode='same')

# Paso 4: Detección de picos
threshold = 0.6 * np.max(ecq_integrated)  # Umbral ajustable
peaks, _ = find_peaks(ecq_integrated, height=threshold, distance=int(0.2 * fs_ecg))



# Graficar señales
plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.plot(ecgSignal)
plt.title("Señal de ECG")

# Señal de ECG filtrada
plt.subplot(3, 1, 2)
plt.plot(filtered_ecg, label="ECG Filtrado (0.1-100 Hz)", color='b')
plt.plot(peaks, filtered_ecg[peaks], 'ro', label="Picos R Detectados")  
plt.title("Señal de ECG Filtrada")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()

# Ventana integrada y detección de picos
plt.subplot(3, 1, 3)
plt.plot(ecq_integrated, label="Ventana Integrada", color='g')
plt.plot(peaks, ecq_integrated[peaks], "x", label="Picos R detectados", color='r')
plt.title("Detección de Picos R - Pan-Tompkins")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()

# Espectro de frecuencia para ECG
#plt.subplot(3, 1, 3)
#plt.plot(fft_frecuencias_ecg[:N_ecg // 2], fft_magnitud_ecg[:N_ecg // 2], color='b')
#plt.title("Espectro de Magnitud FFT para la Señal de ECG")
#plt.xlabel("Frecuencia (Hz)")
#plt.ylabel("Magnitud")
#plt.grid()

plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(eegSignal)
plt.title("Señal de EEG")

# Señal de EEG filtrada
plt.subplot(3, 1, 2)
plt.plot(filtered_eeg, label="EEG Filtrado (0.1-30 Hz)", color='g')
plt.title("Señal de EEG Filtrada")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()

# Espectro de frecuencia para EEG
plt.subplot(3, 1, 3)
plt.plot(fft_frecuencias_eeg[:N_eeg // 2], fft_magnitud_eeg[:N_eeg // 2], color='g')
plt.title("Espectro de Magnitud FFT para la Señal de EEG")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

plt.tight_layout()
plt.show()

#canales_eeg = [
#    "C3", "C4", "O1", "O2", "Cz", "F3", "F4", "F7", "F8", "Fz", "Fp1", "Fp2", "Fpz",
#    "P3", "P4", "Pz", "T3", "T4", "T5", "T6"
#]

# Iterar por cada canal para graficarlo en una figura separada
#for i, canal in enumerate(canales_eeg):
#    plt.figure(figsize=(10, 4))  # Crear una figura independiente
#    plt.plot(eegSignals[i, :])  # Graficar la señal correspondiente
#    plt.title(f"Señal de EEG - Canal {canal}")
#    plt.xlabel("Muestras")
#    plt.ylabel("Amplitud")
#    plt.grid()
#plt.show()
