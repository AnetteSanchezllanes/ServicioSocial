import re
import os
import scipy.io
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GENERAL ===
directorio_principal = r"D:\Mecatronica\Decimo semestre\SS\Paciente0332"
num_archivos_a_procesar = 105
patron_nombre_archivo_mat = re.compile(r".*ECG\.mat$", re.IGNORECASE)  # Asegurar que coincida con tus archivos

archivos_mat = [nombre for nombre in os.listdir(directorio_principal)
                if patron_nombre_archivo_mat.match(nombre)][:num_archivos_a_procesar]

if not archivos_mat:
    print(f"No se encontraron archivos ECG .mat en el directorio: {directorio_principal}")
    exit()
elif len(archivos_mat) < num_archivos_a_procesar:
    print(f"Se encontraron {len(archivos_mat)} archivos ECG .mat. Se procesarán estos archivos.")
else:
    print(f"Se procesarán los primeros {num_archivos_a_procesar} archivos ECG .mat encontrados.")

# Parámetros de muestreo y segmentación
fs = 250
tiempo_inicio = 750  # en segundos
duracion_ventana = 10  # en segundos
muestras_ventana = int(duracion_ventana * fs)
muestras_inicio_archivo = int(tiempo_inicio * fs)

# Almacenamiento de las métricas de VFC a lo largo del tiempo
all_hrv_metrics = {
    'mean_RR': [],
    'std_RR': [],
    'rmssd': [],
}

print("\n=== Procesando archivos y calculando VFC ===")

for idx_archivo, archivo_mat in enumerate(archivos_mat):
    ruta_archivo_mat = os.path.join(directorio_principal, archivo_mat)
    print(f"\n--- Procesando archivo: {archivo_mat} (Hora {idx_archivo + 1}) ---")

    try:
        data = scipy.io.loadmat(ruta_archivo_mat)
        ecg_signal = data['val'][0]  # Asegúrate que 'val' es la clave correcta

        # Filtro pasabanda (0.1–30 Hz)
        b, a = signal.butter(4, [0.1, 30], btype='bandpass', fs=fs)
        ecg_filtrado = signal.filtfilt(b, a, ecg_signal)

        # Ventaneo y cálculo de métricas
        num_ventanas = (len(ecg_filtrado) - muestras_inicio_archivo) // muestras_ventana

        for i in range(num_ventanas):
            inicio = muestras_inicio_archivo + i * muestras_ventana
            fin = inicio + muestras_ventana
            segmento = ecg_filtrado[inicio:fin]

            if len(segmento) < 0.5 * fs:
                continue

            # Detección de picos R
            picos_R, _ = signal.find_peaks(segmento, height=np.mean(segmento), distance=fs/2)

            if len(picos_R) > 1:
                tiempos_pico = np.arange(len(segmento))[picos_R] / fs
                intervalos_RR = np.diff(tiempos_pico) * 1000  # en ms

                if len(intervalos_RR) > 1:
                    mean_RR = np.mean(intervalos_RR)
                    std_RR = np.std(intervalos_RR)
                    rmssd = np.sqrt(np.mean(np.square(np.diff(intervalos_RR))))

                    all_hrv_metrics['mean_RR'].append(mean_RR)
                    all_hrv_metrics['std_RR'].append(std_RR)
                    all_hrv_metrics['rmssd'].append(rmssd)

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_archivo_mat}")
    except KeyError:
        print(f"Error: El archivo {ruta_archivo_mat} no contiene la clave 'val'.")
    except Exception as e:
        print(f"Ocurrió un error al procesar {ruta_archivo_mat}: {e}")

# Estimación del tiempo (en horas) para graficar
duracion_total_horas = num_archivos_a_procesar * duracion_ventana / 3600
tiempo = np.linspace(0, duracion_total_horas, len(all_hrv_metrics['mean_RR']))

# === GRAFICACIÓN ===

plt.figure(figsize=(12, 6))
plt.plot(tiempo, all_hrv_metrics['mean_RR'], label='Media RR (ms)')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Valor')
plt.title('Evolución de la Media de los Intervalos RR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(tiempo, all_hrv_metrics['std_RR'], label='SDNN (ms)')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Valor')
plt.title('Evolución de la Desviación Estándar de RR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(tiempo, all_hrv_metrics['rmssd'], label='RMSSD (ms)')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Valor')
plt.title('Evolución de RMSSD')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Gráficas de VFC generadas ---")
