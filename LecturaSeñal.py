import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch, spectrogram
import pywt

# Cerrar figuras abiertas
plt.close("all")

# Ruta del directorio
folder_path = "D:/Mecatronica/Decimo semestre/SS/Paciente0332"

# Frecuencia de muestreo y duración del segmento (30 segundos en muestras)
sampling_rate = 200  # Frecuencia de muestreo (ajusta según los datos)
segment_duration = 30 * sampling_rate  # 30 segundos

# Número de señales a procesar
num_signals_to_process = 4

# Número de señales a ignorar
num_signals_to_skip = 4

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

# Función para dividir una señal en segmentos de duración específica
def split_signal_into_segments(signal, segment_size):
    num_segments = len(signal) // segment_size
    segments = [signal[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]
    return np.array(segments)

# Función para filtrar y centrar una señal
def preprocess_signal(signal, fs, lowcut, highcut):
    mean_signal = np.mean(signal)
    centered_signal = signal - mean_signal
    filtered_signal = apply_bandpass_filter(centered_signal, lowcut, highcut, fs)
    return filtered_signal

# Calcular y graficar densidad espectral de potencia
def plot_power_spectral_density(signal, fs, label, ax):
    window_size = 200  # Tamaño de la ventana en datos
    overlap = int(0.75 * window_size)  # Traslape del 75%
    f, Pxx = welch(signal, fs=fs, window='hamming', nperseg=window_size, noverlap=overlap)
    ax.semilogy(f, Pxx, label=label)  # Gráfica en escala logarítmica
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Densidad espectral de potencia [V²/Hz]")
    ax.grid()

def plot_spectrogram(signal, fs, label, ax):
    f, t, Sxx = spectrogram(signal, fs=fs, window='hamming', nperseg=200, noverlap=150)
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Frecuencia [Hz]")
    ax.set_title(f"Espectrograma - {label}")
    # Añadir solo una barra de color para el espectrograma
    if not hasattr(ax, 'colorbar_added'):  # Verificar si la barra ya fue añadida
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Densidad de potencia [dB]")
        ax.colorbar_added = True  # Marcar que la barra de color ya ha sido añadida


# Solicitar al usuario seleccionar los canales deseados
selected_ecg_channel = 1
selected_eeg_channel = 1

# Listas para almacenar señales procesadas
ecgs_to_plot = []
eegs_to_plot = []

# Contador para limitar el número de señales procesadas
signals_processed = 0
signals_skipped = 0

# Procesar archivos .mat en la carpeta
for file_name in os.listdir(folder_path):
    if file_name.endswith(".mat") and "REF" not in file_name and "OTHER" not in file_name:
        file_path = os.path.join(folder_path, file_name)

        if signals_skipped < num_signals_to_skip:
            signals_skipped += 1
            continue  # Ignorar las primeras señales

        print(f"Procesando archivo: {file_name}")

        # Clasificar archivo como ECG o EEG
        is_ecg = "ECG" in file_name.upper()
        is_eeg = "EEG" in file_name.upper()

        # Cargar datos
        data = loadmat(file_path)

        if "val" in data:
            signals = np.array(data["val"])

            # Procesar señal ECG
            if is_ecg and selected_ecg_channel < signals.shape[0]:
                ecg_signal = signals[selected_ecg_channel]
                filtered_ecg = preprocess_signal(ecg_signal, fs=sampling_rate, lowcut=0.1, highcut=60)
                ecg_segments = split_signal_into_segments(filtered_ecg, segment_duration)

                if len(ecg_segments) > 0:
                    ecgs_to_plot.append(ecg_segments[0])

            # Procesar señal EEG
            elif is_eeg and selected_eeg_channel < signals.shape[0]:
                eeg_signal = signals[selected_eeg_channel]
                filtered_eeg = preprocess_signal(eeg_signal, fs=sampling_rate, lowcut=0.1, highcut=30)
                eeg_segments = split_signal_into_segments(filtered_eeg, segment_duration)

                if len(eeg_segments) > 0:
                    eegs_to_plot.append(eeg_segments[0])

            signals_processed += 1
            if signals_processed >= num_signals_to_process:
                print("Se alcanzó el límite de señales a procesar.")
                break
        else:
            print(f"El archivo {file_name} no contiene datos bajo la clave 'val'.")

    if signals_processed >= num_signals_to_process:
        break

# Graficar señales, PSD y espectrograma para ECG
if ecgs_to_plot:
    fig_ecg, axs_ecg = plt.subplots(3, 1, figsize=(12, 9))
    fig_ecg.suptitle("Señales ECG, su PSD y Espectrograma")

    axs_ecg[0].set_title("Señales ECG")
    for i, ecg in enumerate(ecgs_to_plot):
        axs_ecg[0].plot(ecg, label=f"ECG {i + 1}")
        plot_power_spectral_density(ecg, sampling_rate, f"ECG {i + 1}", axs_ecg[1])
        plot_spectrogram(ecg, sampling_rate, f"ECG {i + 1}", axs_ecg[2])
    axs_ecg[0].set_xlabel("Muestras")
    axs_ecg[0].set_ylabel("Amplitud")
    axs_ecg[0].legend()

    axs_ecg[1].set_title("PSD - Señales ECG")
    axs_ecg[2].set_title("Espectrograma - Señales ECG")
    plt.tight_layout()
    plt.show()

# Graficar señales, PSD y espectrograma para EEG
if eegs_to_plot:
    fig_eeg, axs_eeg = plt.subplots(3, 1, figsize=(12, 9))
    fig_eeg.suptitle("Señales EEG, su PSD y Espectrograma")

    axs_eeg[0].set_title("Señales EEG")
    for i, eeg in enumerate(eegs_to_plot):
        axs_eeg[0].plot(eeg, label=f"EEG {i + 1}")
        plot_power_spectral_density(eeg, sampling_rate, f"EEG {i + 1}", axs_eeg[1])
        plot_spectrogram(eeg, sampling_rate, f"EEG {i + 1}", axs_eeg[2])
    axs_eeg[0].set_xlabel("Muestras")
    axs_eeg[0].set_ylabel("Amplitud")
    axs_eeg[0].legend()

    axs_eeg[1].set_title("PSD - Señales EEG")
    axs_eeg[2].set_title("Espectrograma - Señales EEG")
    plt.tight_layout()
    plt.show()

print("Procesamiento completado.")
