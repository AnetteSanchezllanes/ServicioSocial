import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

# Leer archivo .mat
ruta = r"D:/Mecatronica/Decimo semestre/SS/Paciente0332/0332_003_024_ECG.mat"
data = sio.loadmat(ruta)
print("Claves en el archivo .mat:", data.keys())

# Extraer la señal de ECG 
ecg_signal = data['val'][0]

# Frecuencia de muestreo 
fs = 200 # 200Hz

# Crear el eje de tiempo en segundos
t = np.arange(len(ecg_signal)) / fs

# Filtrar la señal (pasa bandas entre 0.1 y 50 Hz)
b, a = signal.butter(4, [0.1, 50], btype='bandpass', fs=fs)
ecg_filtrado = signal.filtfilt(b, a, ecg_signal)

# Detectar los picos R
picos_R, _ = signal.find_peaks(ecg_filtrado, height=np.mean(ecg_filtrado), distance=fs/2)

# Graficar el ECG con los picos R detectados
plt.figure(figsize=(10, 4))
plt.plot(t, ecg_filtrado, label="ECG Filtrado")
plt.plot(t[picos_R], ecg_filtrado[picos_R], 'ro', label="Picos R")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.title("Detección de Picos R")
plt.show()

# Calcular intervalos RR
tiempos_R = t[picos_R]
intervalos_RR = np.diff(tiempos_R) * 1000  # Convertir a milisegundos

# Graficar los intervalos RR
plt.figure(figsize=(8, 3))
plt.plot(intervalos_RR, marker='o', linestyle='-', label="Intervalos RR (ms)")
plt.ylabel("Tiempo entre latidos (ms)")
plt.xlabel("Latido")
plt.legend()
plt.title("Variabilidad de la Frecuencia Cardíaca (VFC)")
plt.show()

# Función para calcular métricas de HRV en el dominio del tiempo
def calcular_hrv_tiempo(intervalos_RR):
    resultados = {}
    resultados['mean_RR'] = np.mean(intervalos_RR)  # Media de los intervalos RR
    resultados['std_RR'] = np.std(intervalos_RR)    # Desviación estándar de los intervalos RR
    resultados['rmssd'] = np.sqrt(np.mean(np.square(np.diff(intervalos_RR))))  # RMSSD
    resultados['nn50'] = np.sum(np.abs(np.diff(intervalos_RR)) > 50)  # Número de pares de intervalos RR consecutivos que difieren en más de 50 ms
    resultados['pnn50'] = (resultados['nn50'] / len(intervalos_RR)) * 100  # Porcentaje de nn50
    return resultados

# Función para calcular métricas de HRV en el dominio de la frecuencia
def calcular_hrv_frecuencia(intervalos_RR, fs=4.0):
    # Interpolación para obtener una señal uniformemente muestreada
    tiempo_RR = np.cumsum(intervalos_RR) / 1000.0  # Convertir a segundos
    tiempo_interp = np.arange(tiempo_RR[0], tiempo_RR[-1], 1/fs)
    rr_interp = np.interp(tiempo_interp, tiempo_RR, intervalos_RR)

    # Calcular la transformada de Fourier
    n = len(rr_interp)
    frecuencias = np.fft.rfftfreq(n, d=1/fs)
    espectro = np.abs(np.fft.rfft(rr_interp - np.mean(rr_interp)))

    # Bandas de frecuencia
    vlf_band = (0.0033, 0.04)  # Muy baja frecuencia
    lf_band = (0.04, 0.15)     # Baja frecuencia
    hf_band = (0.15, 0.4)      # Alta frecuencia

    # Calcular la potencia en cada banda
    vlf = np.trapz(espectro[(frecuencias >= vlf_band[0]) & (frecuencias < vlf_band[1])], 
                   frecuencias[(frecuencias >= vlf_band[0]) & (frecuencias < vlf_band[1])])
    lf = np.trapz(espectro[(frecuencias >= lf_band[0]) & (frecuencias < lf_band[1])], 
                  frecuencias[(frecuencias >= lf_band[0]) & (frecuencias < lf_band[1])])
    hf = np.trapz(espectro[(frecuencias >= hf_band[0]) & (frecuencias < hf_band[1])], 
                  frecuencias[(frecuencias >= hf_band[0]) & (frecuencias < hf_band[1])])

    # Calcular la relación LF/HF
    lf_hf_ratio = lf / hf

    resultados = {
        'vlf': vlf,
        'lf': lf,
        'hf': hf,
        'lf_hf_ratio': lf_hf_ratio
    }
    return resultados

# Calcular métricas de HRV manualmente
resultados_tiempo = calcular_hrv_tiempo(intervalos_RR)
resultados_frecuencia = calcular_hrv_frecuencia(intervalos_RR)

# Mostrar resultados
print("📊 Análisis en el Dominio del Tiempo:")
print(resultados_tiempo)

print("\n📊 Análisis en el Dominio de la Frecuencia:")
print(resultados_frecuencia)