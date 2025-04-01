import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

# 1. Configuración inicial
fs = 200  # Frecuencia de muestreo (Hz)
ruta_archivo = r"D:\Mecatronica\Decimo semestre\SS\Paciente0332\0332_003_024_ECG.mat"  # ¡Ajusta esta ruta!

try:
    # 2. Cargar y verificar datos
    data = sio.loadmat(ruta_archivo)
    print("Claves en el archivo:", data.keys())  # Verificar estructura
    
    # Extraer señal  
    ecg_signal = data['val'][0]  
    print(f"Rango inicial: {np.min(ecg_signal):.6f} a {np.max(ecg_signal):.6f} (¿mV o V?)")
    
    # Ajustar escala
    ecg_signal = ecg_signal * 1000  # Si estaba en V, ahora en mV
    
    # 3. Segmento de interés (100-220 segundos)
    inicio, fin = 100 * fs, 220 * fs
    ecg_segmento = ecg_signal[inicio:fin]
    t = np.arange(inicio, fin) / fs
    
    # 4. Filtrado (Notch + Pasa-bandas)
    # Eliminar 50/60 Hz (ruido eléctrico)
    b_notch, a_notch = signal.iirnotch(50, 30, fs)
    ecg_notch = signal.filtfilt(b_notch, a_notch, ecg_segmento)
    
    # Filtrar frecuencias no cardíacas
    b, a = signal.butter(4, [0.5, 50], btype='bandpass', fs=fs)  # Ajustado a 0.5 Hz para eliminar línea base
    ecg_filtrado = signal.filtfilt(b, a, ecg_notch)
    
    # 5. Detección de picos R (parámetros optimizados)
    picos_R, _ = signal.find_peaks(
        ecg_filtrado,
        height=np.median(ecg_filtrado) * 3,  # Umbral alto para evitar ruido
        distance=fs * 0.6,  # Distancia mínima entre latidos
        prominence=np.std(ecg_filtrado) * 4  # Evitar picos pequeños
    )
    
    # 6. Calcular intervalos RR y filtrar
    tiempos_R = t[picos_R]
    intervalos_RR = np.diff(tiempos_R) * 1000  # Convertir a ms
    intervalos_RR = intervalos_RR[(intervalos_RR > 300) & (intervalos_RR < 1500)]  # Rango fisiológico
    
    # 7. Gráficos de diagnóstico
    plt.figure(figsize=(15, 5))
    
    # Señal ECG + picos R
    plt.subplot(1, 2, 1)
    plt.plot(t, ecg_filtrado, label="ECG filtrado", linewidth=1)
    plt.plot(t[picos_R], ecg_filtrado[picos_R], 'ro', markersize=5, label="Picos R")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (mV)")
    plt.title("Señal ECG con Picos R Detectados")
    plt.legend()
    plt.grid()
    
    # Boxplot de intervalos RR (solo si hay datos)
    plt.subplot(1, 2, 2)
    if len(intervalos_RR) > 0:
        plt.boxplot(
            intervalos_RR,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red'),
            widths=0.5
        )
        plt.title("Distribución de Intervalos RR")
        plt.ylabel("Intervalo RR (ms)")
        plt.ylim(300, 1200)  # Rango fisiológico ajustado
    else:
        plt.text(0.5, 0.5, "No hay intervalos RR válidos", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # 8. Métricas de HRV
    if len(intervalos_RR) > 1:
        print("\n📊 Métricas de HRV:")
        print(f"- Latidos detectados: {len(intervalos_RR)}")
        print(f"- Media RR: {np.mean(intervalos_RR):.2f} ms")
        print(f"- RMSSD: {np.sqrt(np.mean(np.square(np.diff(intervalos_RR)))):.2f} ms (variabilidad cardíaca)")
    else:
        print("⚠️ No hay suficientes latidos para calcular HRV.")

except Exception as e:
    print(f"❌ Error: {str(e)}")