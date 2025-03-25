import os
import numpy as np
from scipy.io import loadmat

# Ruta de la carpeta "Pacientes"
carpeta_pacientes = "Pacientes"

# Función para leer archivos .hea y extraer información
def leer_archivo_hea(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lineas = archivo.readlines()
        primera_linea = lineas[0].strip().split()
        frecuencia_muestreo = int(primera_linea[2])
        numero_muestras = int(primera_linea[3])
        return frecuencia_muestreo, numero_muestras

# Función para contar la cantidad de datos en un archivo .mat
def contar_datos_mat(ruta_archivo):
    try:
        datos = loadmat(ruta_archivo)
        # Buscar la clave "val" que contiene los datos
        if "val" in datos:
            return datos["val"].shape[1]  # Asumiendo que los datos están en la segunda dimensión
        else:
            print(f"Advertencia: No se encontró la clave 'val' en {ruta_archivo}")
            return 0
    except Exception as e:
        print(f"Error al leer el archivo {ruta_archivo}: {e}")
        return 0

# Recorrer la carpeta de pacientes
for paciente in os.listdir(carpeta_pacientes):
    ruta_paciente = os.path.join(carpeta_pacientes, paciente)
    
    # Verificar si es una carpeta
    if os.path.isdir(ruta_paciente):
        print(f"\nPaciente: {paciente}")
        
        # Buscar archivos .hea y .mat
        archivos_hea = [f for f in os.listdir(ruta_paciente) if f.endswith('.hea')]
        archivos_mat = [f for f in os.listdir(ruta_paciente) if f.endswith('.mat')]
        
        # Procesar archivos .hea
        for archivo_hea in archivos_hea:
            ruta_hea = os.path.join(ruta_paciente, archivo_hea)
            frecuencia_muestreo_hea, numero_muestras = leer_archivo_hea(ruta_hea)
            print(f"  Archivo .hea: {archivo_hea}")
            print(f"    Frecuencia de muestreo (declarada): {frecuencia_muestreo_hea} Hz")
            print(f"    Número de muestras esperado: {numero_muestras}")
        
        # Procesar archivos .mat
        for archivo_mat in archivos_mat:
            ruta_mat = os.path.join(ruta_paciente, archivo_mat)
            cantidad_datos = contar_datos_mat(ruta_mat)
            print(f"  Archivo .mat: {archivo_mat}")
            print(f"    Número de muestras en el archivo: {cantidad_datos}")
            
            # Verificar consistencia entre .hea y .mat
            if cantidad_datos == numero_muestras:
                print("    ✅ Los datos son consistentes.")
            else:
                print("    ❌ Los datos NO son consistentes.")
            
            # Calcular la duración del registro
            if cantidad_datos > 0:
                duracion = cantidad_datos / frecuencia_muestreo_hea
                print(f"    Duración del registro: {duracion} segundos")
                
                # Calcular la frecuencia de muestreo a partir de los datos
                frecuencia_muestreo_calculada = cantidad_datos / duracion
                print(f"    Frecuencia de muestreo (calculada): {frecuencia_muestreo_calculada} Hz")
                
                # Comparar la frecuencia declarada con la calculada
                if np.isclose(frecuencia_muestreo_hea, frecuencia_muestreo_calculada, rtol=1e-3):
                    print("    ✅ La frecuencia de muestreo es correcta.")
                else:
                    print("    ❌ La frecuencia de muestreo NO es correcta.")
            else:
                print("    Duración del registro: No disponible (archivo vacío o corrupto).")