import scipy.io
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# === CONFIGURACIÓN GENERAL ===
directorio_principal = "D:/Mecatronica/Decimo semestre/SS/Paciente0332/"
num_archivos_a_procesar = 30
archivos_por_grafica = 5
patron_nombre_archivo = re.compile(r".*EEG\.mat$", re.IGNORECASE)

archivos_eeg = []
for nombre_archivo in os.listdir(directorio_principal):
    if patron_nombre_archivo.match(nombre_archivo):
        archivos_eeg.append(nombre_archivo)

archivos_a_procesar = archivos_eeg[:num_archivos_a_procesar]

if not archivos_a_procesar:
    print(f"No se encontraron archivos EEG .mat en el directorio: {directorio_principal}")
    exit()
elif len(archivos_a_procesar) < num_archivos_a_procesar:
    print(f"Se encontraron {len(archivos_a_procesar)} archivos EEG .mat. Se procesarán estos archivos.")
else:
    print(f"Se procesarán los primeros {num_archivos_a_procesar} archivos EEG .mat encontrados.")

# Parámetros de muestreo y segmentación
fs = 250
tiempo_inicio = 120
duracion_ventana = 3
muestras_inicio = int(tiempo_inicio * fs)
muestras_fin = muestras_inicio + int(duracion_ventana * fs)

# Canal deseado
canal_deseado = 4

# Wavelet y bandas
wavelet = 'db4'
wavelet_obj = pywt.Wavelet(wavelet)
bandas = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma Baja', 'Gamma Alta']
colores = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

# === PROCESO Y GRAFICACIÓN POR LOTES ===
num_lotes = (len(archivos_a_procesar) + archivos_por_grafica - 1) // archivos_por_grafica

for lote in range(num_lotes):
    start_index = lote * archivos_por_grafica
    end_index = min((lote + 1) * archivos_por_grafica, len(archivos_a_procesar))
    batch_archivos = archivos_a_procesar[start_index:end_index]
    num_archivos_en_batch = len(batch_archivos)

    plt.figure(figsize=(14, 12 * num_archivos_en_batch // 2))

    for idx_en_lote, archivo in enumerate(batch_archivos):
        ruta_archivo = os.path.join(directorio_principal, archivo)
        try:
            data = scipy.io.loadmat(ruta_archivo)
            eeg = data['val']
            canal = eeg[canal_deseado, :]
            segmento = canal[muestras_inicio:muestras_fin]
            nivel_max = pywt.dwt_max_level(len(segmento), wavelet_obj.dec_len)
            coef = pywt.wavedec(segmento, wavelet, level=nivel_max)
            asignaciones = dict(zip(bandas, list(range(nivel_max, nivel_max - len(bandas), -1))))
            bandas_disponibles = list(asignaciones.keys())
            reconstruidas = []
            for banda in bandas_disponibles:
                nivel = asignaciones[banda]
                coef_band = [np.zeros_like(c) for c in coef]
                coef_band[nivel] = coef[nivel]
                rec = pywt.waverec(coef_band, wavelet)
                reconstruidas.append(rec[:len(segmento)])

            # Graficar bandas
            tiempo = np.arange(len(segmento)) / fs
            plt.subplot(num_archivos_en_batch, 1, idx_en_lote + 1)
            for b_idx, (rec, banda) in enumerate(zip(reconstruidas, bandas_disponibles)):
                plt.plot(tiempo, rec, color=colores[b_idx], label=f'Banda {banda}')

            plt.title(f"Bandas EEG para {archivo} - Canal {canal_deseado + 1}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud")
            plt.legend(fontsize='small')

            # Cálculo y muestra de energías (solo para el primer lote)
            if lote == 0 and idx_en_lote == 0:
                print("\n=== Energía por Banda para los primeros archivos ===")
            if lote == 0:
                energias = [np.sum(b**2) for b in reconstruidas]
                print(f"\n--- {archivo} (Canal {canal_deseado + 1}) ---")
                for b_idx, energia in enumerate(energias):
                    print(f"{bandas_disponibles[b_idx]:12s}: {energia:.2f}")

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {ruta_archivo}")
        except KeyError:
            print(f"Error: El archivo {ruta_archivo} no contiene la clave 'val'.")
        except Exception as e:
            print(f"Ocurrió un error al procesar {ruta_archivo}: {e}")

    plt.tight_layout()

# Mostrar todas las figuras al final
plt.show()

# Mostrar energías para los archivos restantes después del primer lote
if num_lotes > 1:
    print("\n=== Energía por Banda para los archivos restantes ===")
    for lote in range(1, num_lotes):
        start_index = lote * archivos_por_grafica
        end_index = min((lote + 1) * archivos_por_grafica, len(archivos_a_procesar))
        batch_archivos = archivos_a_procesar[start_index:end_index]
        for archivo in batch_archivos:
            ruta_archivo = os.path.join(directorio_principal, archivo)
            try:
                data = scipy.io.loadmat(ruta_archivo)
                eeg = data['val']
                canal = eeg[canal_deseado, :]
                segmento = canal[muestras_inicio:muestras_fin]
                nivel_max = pywt.dwt_max_level(len(segmento), wavelet_obj.dec_len)
                coef = pywt.wavedec(segmento, wavelet, level=nivel_max)
                asignaciones = dict(zip(bandas, list(range(nivel_max, nivel_max - len(bandas), -1))))
                bandas_disponibles = list(asignaciones.keys())
                reconstruidas = []
                energias = [np.sum(b**2) for b in [pywt.waverec([np.zeros_like(c) for c in coef[:l]] + [coef[l]] + [np.zeros_like(c) for c in coef[l+1:]], wavelet)[:len(segmento)]
                                                   for l in asignaciones.values()]]
                print(f"\n--- {archivo} (Canal {canal_deseado + 1}) ---")
                for b_idx, energia in enumerate(energias):
                    print(f"{bandas_disponibles[b_idx]:12s}: {energia:.2f}")
            except Exception as e:
                print(f"Ocurrió un error al calcular energías para {archivo}: {e}")