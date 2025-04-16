import scipy.io
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

fs = 512  # Frecuencia de muestreo en Hz
wavelet = 'db4'
niveles = 6
bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma Baja', 'Gamma Alta']
asignaciones = {banda: i+1 for i, banda in enumerate(reversed(bandas))}  # Asignación automática
num_canales = 5  # Número de canales a analizar 
base_path = r"D:/Mecatronica/Decimo semestre/SS/Paciente0332/"
archivos = []

for i in range(5, 9):
    for j in range(24, 28):
        archivo = f"0332_{i:03d}_{j:03d}_EEG.mat"
        ruta = os.path.join(base_path, archivo)
        if os.path.exists(ruta):
            archivos.append(ruta)
        else:
            print(f"⚠️ Archivo no encontrado: {archivo}")

def procesar_eeg(ruta, canal):
    try:
        data = scipy.io.loadmat(ruta)
        señal = data['val'][canal, :]
        
        # Normalización profesional
        señal = (señal - np.mean(señal)) / np.std(señal)
        
        coef = pywt.wavedec(señal, wavelet, level=niveles)
        reconstrucciones = []
        
        for banda in bandas:
            nivel = asignaciones[banda]
            coef_temp = [np.zeros_like(c) for c in coef]
            coef_temp[nivel] = coef[nivel]
            rec = pywt.waverec(coef_temp, wavelet)[:len(señal)]
            reconstrucciones.append(rec)
            
        return señal, reconstrucciones
    
    except Exception as e:
        print(f"❌ Error en {os.path.basename(ruta)} (Canal {canal}): {str(e)}")
        return None, None

for archivo in archivos:
    print(f"\n🔍 Procesando: {os.path.basename(archivo)}")
    
    # Crear una figura para este archivo (5 canales × 2 columnas)
    plt.figure(figsize=(20, 6 * num_canales))
    gs = GridSpec(num_canales, 2, figure=plt.gcf())
    plt.suptitle(f"Análisis EEG: {os.path.basename(archivo)}", y=1.02, fontsize=14)
    
    for canal in range(num_canales):
        señal, reconstrucciones = procesar_eeg(archivo, canal)
        if señal is None:
            continue
        
        # Crear vector de tiempo en segundos
        tiempo = np.arange(len(señal)) / fs
        
        # Columna izquierda: Señal original
        ax0 = plt.subplot(gs[canal, 0])
        ax0.plot(tiempo, señal, color='#1f77b4', linewidth=0.8)
        ax0.set_title(f"Canal {canal + 1} - EEG Original", pad=10)
        ax0.set_xlabel('Tiempo (s)')
        ax0.set_ylabel('Amplitud (uV)')
        ax0.grid(True, alpha=0.3)
        
        # Columna derecha: Bandas
        ax1 = plt.subplot(gs[canal, 1])
        colors = plt.cm.viridis(np.linspace(0, 1, len(bandas)))
        
        for i, (banda, color) in enumerate(zip(bandas, colors)):
            ax1.plot(tiempo, reconstrucciones[i], color=color, label=banda, linewidth=0.7)
        
        ax1.set_title(f"Canal {canal + 1} - Descomposición Wavelet", pad=10)
        ax1.set_xlabel('Tiempo (s)')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True, alpha=0.3)
        
        # Ajustes estéticos
        for ax in [ax0, ax1]:
            ax.set_xlim(0, tiempo[-1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

print("\n✅ Procesamiento completado")