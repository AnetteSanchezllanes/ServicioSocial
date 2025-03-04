import os
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def extract_patient_data(file_path):
    """Extrae los valores de Sexo y Edad de un archivo de texto."""
    sex, age = None, None
    patient_id = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Patient:"):
                patient_id = line.split(":")[1].strip()
            elif line.startswith("Sex:"):
                sex = line.split(":")[1].strip()
            elif line.startswith("Age:"):
                age = line.split(":")[1].strip()
    return patient_id, sex if sex and sex.lower() != "nan" else None, age if age and age.isdigit() else None

def analyze_patients_data(base_dir):
    """Recorre las carpetas y obtiene la cantidad de hombres, mujeres y distribución de edades."""
    sex_count = defaultdict(int)
    age_list = []
    patients_with_nan_sex = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):  # Solo procesamos archivos de texto
                file_path = os.path.join(root, file)
                patient_id, sex, age = extract_patient_data(file_path)
                if sex:
                    sex_count[sex] += 1
                else:
                    if patient_id:
                        patients_with_nan_sex.append(patient_id)
                if age:
                    age_list.append(int(age))
    
    return sex_count, age_list, patients_with_nan_sex

def plot_pie_chart(data, title):
    """Genera una gráfica circular con los datos proporcionados."""
    labels = list(data.keys())
    sizes = list(data.values())
    colors = ['lightblue', 'pink']
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, shadow=True)
    plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_bar_chart(data, title, xlabel, ylabel):
    """Genera una gráfica de barras con los datos proporcionados."""
    age_counts = Counter(data)
    sorted_ages = sorted(age_counts.keys())
    counts = [age_counts[age] for age in sorted_ages]
    
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_ages, counts, color='green')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    

if __name__ == "__main__":
    base_directory = r"D:\Mecatronica\Decimo semestre\SS\Datos txt\physionet.org\files\i-care\2.0\training"
    sex_data, age_data, nan_sex_patients = analyze_patients_data(base_directory)
    
    print("Cantidad de pacientes por género:")
    for sex, count in sex_data.items():
        print(f"{sex}: {count} Pacientes")
    
    print("\nEdades con mayor frecuencia:")
    age_counter = Counter(age_data)
    most_common_ages = age_counter.most_common(10)
    for age, count in most_common_ages:
        print(f"Edad {age}: {count} Pacientes")
    
    # Generar gráficas
    plot_pie_chart(sex_data, "Distribución por género")
    plot_bar_chart(age_data, "Distribución de edades", "Edad", "Cantidad de Pacientes")
    
    print("\nPacientes con valor 'nan' en género:")
    for patient in nan_sex_patients:
        print(f"Paciente ID: {patient}")
plt.show()