import re
import os
import scipy.io
import pywt
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GENERAL ===
directorio_principal = r"D:\Mecatronica\Decimo semestre\SS\Paciente0332"
num_archivos_a_procesar = 24  # Procesando las 24 horas

patron_nombre_archivo_mat = re.compile(r".*EEG\.mat$", re.IGNORECASE)
patron_nombre_archivo_hea = re.compile(r".*\.hea$", re.IGNORECASE)

archivos_mat = []
archivos_hea = {}

for nombre_archivo in os.listdir(directorio_principal):
    if patron_nombre_archivo_mat.match(nombre_archivo):
        archivos_mat.append(nombre_archivo)
    elif patron_nombre_archivo_hea.match(nombre_archivo):
        nombre_base = os.path.splitext(nombre_archivo)[0]
        archivos_hea[nombre_base] = os.path.join(directorio_principal, nombre_archivo)

archivos_a_procesar = archivos_mat[:num_archivos_a_procesar]

if not archivos_a_procesar:
    print(f"No se encontraron archivos EEG .mat en el directorio: {directorio_principal}")
    exit()
elif len(archivos_a_procesar) < num_archivos_a_procesar:
    print(f"Se encontraron {len(archivos_a_procesar)} archivos EEG .mat. Se procesarán estos archivos.")
else:
    print(f"Se procesarán los primeros {num_archivos_a_procesar} archivos EEG .mat encontrados.")

# Parámetros de muestreo y segmentación
fs = 250
tiempo_inicio = 1000
duracion_ventana = 3
muestras_inicio = int(tiempo_inicio * fs)
muestras_fin = muestras_inicio + int(duracion_ventana * fs)

# Número de canales a procesar
num_canales = 19

# === CONFIGURACIÓN DEL FILTRO PASABANDAS ===
frecuencia_corte_bajo = 0.1  # Hz
frecuencia_corte_alto = 30.0  # Hz
orden_filtro = 4

# Wavelet y bandas
wavelet = 'db4'
wavelet_obj = pywt.Wavelet(wavelet)
bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma Baja', 'Gamma Alta']
colores_viridis = plt.cm.viridis(np.linspace(0, 1, num_canales))
colores_fijos = ['red', 'blue', 'green', 'orange', 'black']

# Almacenamiento de energías por banda y canal
all_energies = {banda: {canal: [] for canal in range(num_canales)} for banda in bandas}
canal_nombres_dict = {}

print("\n=== Procesando archivos y extrayendo energías ===")

for idx_archivo, archivo_mat in enumerate(archivos_a_procesar):
    ruta_archivo_mat = os.path.join(directorio_principal, archivo_mat)
    nombre_base = os.path.splitext(archivo_mat)[0]
    ruta_archivo_hea = archivos_hea.get(nombre_base)
    canal_nombres = {}

    # Leer nombres de canales desde archivo .hea
    if ruta_archivo_hea:
        try:
            with open(ruta_archivo_hea, 'r') as f:
                for i, linea in enumerate(f):
                    if i == 0:
                        partes = linea.split()
                        num_canales_hea = int(partes[1]) if len(partes) > 1 else 0
                    elif 0 < i <= num_canales:
                        partes = linea.split()
                        if len(partes) > 4:
                            canal_index = i
                            canal_nombre = partes[-1].strip()
                            canal_nombres[canal_index - 1] = canal_nombre
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo .hea: {ruta_archivo_hea}")
        except Exception as e:
            print(f"Error al leer el archivo .hea {ruta_archivo_hea}: {e}")
    else:
        print(f"Advertencia: No se encontró el archivo .hea para {archivo_mat}")

    canal_nombres_dict[nombre_base] = canal_nombres

    print(f"\n--- Procesando archivo: {archivo_mat} (Hora {idx_archivo + 1}) ---")

    try:
        data = scipy.io.loadmat(ruta_archivo_mat)
        eeg = data['val']

        for canal_idx in range(min(num_canales, eeg.shape[0])):
            canal_original = eeg[canal_idx, :]
            segmento_original = canal_original[muestras_inicio:muestras_fin]

            # Filtro pasabanda
            nyquist_frecuencia = 0.5 * fs
            frecuencia_normalizada_bajo = frecuencia_corte_bajo / nyquist_frecuencia
            frecuencia_normalizada_alto = frecuencia_corte_alto / nyquist_frecuencia
            b, a = butter(orden_filtro, [frecuencia_normalizada_bajo, frecuencia_normalizada_alto], btype='band')
            segmento_filtrado = filtfilt(b, a, segmento_original)

            # Análisis wavelet
            nivel_max_filt = pywt.dwt_max_level(len(segmento_filtrado), wavelet_obj.dec_len)
            coef_filt = pywt.wavedec(segmento_filtrado, wavelet, level=nivel_max_filt)
            asignaciones_filt = dict(zip(bandas, list(range(nivel_max_filt, nivel_max_filt - len(bandas), -1))))
            bandas_disponibles_filt = list(asignaciones_filt.keys())
            energias_filt = [
                np.sum(pywt.waverec(
                    [np.zeros_like(c) for c in coef_filt[:l]] +
                    [coef_filt[l]] +
                    [np.zeros_like(c) for c in coef_filt[l+1:]],
                    wavelet)[:len(segmento_filtrado)] ** 2)
                for l in asignaciones_filt.values()
            ]

            for i, banda in enumerate(bandas_disponibles_filt):
                all_energies[banda][canal_idx].append(energias_filt[i])

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_archivo_mat}")
    except KeyError:
        print(f"Error: El archivo {ruta_archivo_mat} no contiene la clave 'val'.")
    except Exception as e:
        print(f"Ocurrió un error al procesar {ruta_archivo_mat}: {e}")

print("\n=== Generando gráficas ===")

def generar_figura(canales_a_graficar, fig_num):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharex=True, sharey=False)
    axes = axes.flatten()
    start_canal = canales_a_graficar[0] + 1
    end_canal = canales_a_graficar[-1] + 1
    fig.suptitle(f'Energía de las Bandas a lo largo de 24 Horas - Canales {start_canal} a {end_canal}', fontsize=16)

    for idx_banda, banda in enumerate(bandas):
        ax = axes[idx_banda]
        for i, canal_idx in enumerate(canales_a_graficar):
            nombres_archivo = list(canal_nombres_dict.keys())[0] if canal_nombres_dict else None
            nombre_canal = canal_nombres_dict.get(nombres_archivo, {}).get(canal_idx, f"Canal {canal_idx + 1}")
            color = colores_viridis[canal_idx] if fig_num == 1 else colores_fijos[i % len(colores_fijos)]

            y_vals = all_energies[banda][canal_idx]
            x_vals = range(len(y_vals))  # Ajuste dinámico
            ax.plot(x_vals, y_vals, color=color, label=nombre_canal)

        ax.set_title(f'Banda {banda}')
        ax.set_ylabel('Energía')
        ax.grid(True)
        ax.legend(fontsize='small', loc='upper right')
        if idx_banda in [3, 4, 5]:
            ax.set_xlabel('Hora (Archivo)')
            ax.set_xticks(range(num_archivos_a_procesar))
            ax.set_xticklabels([f'{i+1}' for i in range(num_archivos_a_procesar)])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# Generar gráficas
fig1 = generar_figura(range(num_canales), 1)
fig2 = generar_figura(range(5), 2)
fig3 = generar_figura(range(5, 10), 3)
fig4 = generar_figura(range(10, 15), 4)
fig5 = generar_figura(range(15, num_canales), 5)

plt.show()

print("\n--- Gráficas generadas ---")
