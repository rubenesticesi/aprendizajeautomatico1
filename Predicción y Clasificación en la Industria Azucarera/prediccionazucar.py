# Paso 1: Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración para un mejor estilo en los gráficos
sns.set(style="whitegrid")

# --- ANÁLISIS DEL DATASET: HISTORICO_SUERTES.xlsx ---

print("="*50)
print("Análisis del archivo: HISTORICO_SUERTES.xlsx")
print("="*50)

# Paso 2: Cargar el primer conjunto de datos
try:
    df_historico = pd.read_csv('HISTORICO_SUERTES.xlsx - Hoja1.csv')
    print("Archivo 'HISTORICO_SUERTES.xlsx - Hoja1.csv' cargado exitosamente.")
    print(f"El dataset tiene {df_historico.shape[0]} filas y {df_historico.shape[1]} columnas.\n")

    # Paso 3: Análisis exploratorio de datos (EDA)

    # Revisar las primeras filas
    print("--- Primeras 5 filas del dataset ---")
    print(df_historico.head())
    print("\n")

    # Obtener información general y tipos de datos
    print("--- Información general del dataset ---")
    df_historico.info()
    print("\n")

    # Identificar valores faltantes
    print("--- Porcentaje de valores faltantes (Top 10) ---")
    missing_values = df_historico.isnull().sum() / len(df_historico) * 100
    print(missing_values[missing_values > 0].sort_values(ascending=False).head(10))
    print("\n")

    # Estadísticas descriptivas de las variables de interés
    print("--- Estadísticas descriptivas para TCH y %Sac.Caña ---")
    print(df_historico[['TCH', '%Sac.Caña']].describe())
    print("\n")

    # Paso 4: Visualización de TCH y %Sac.Caña

    # Visualización para TCH
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_historico['TCH'], kde=True, bins=30)
    plt.title('Distribución de TCH (Toneladas de Caña/Hectárea)')
    plt.xlabel('TCH')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_historico['TCH'])
    plt.title('Boxplot de TCH para detectar outliers')
    plt.xlabel('TCH')
    plt.tight_layout()
    plt.show()

    # Visualización para %Sac.Caña (eliminando valores nulos para el gráfico)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_historico['%Sac.Caña'].dropna(), kde=True, bins=30, color='green')
    plt.title('Distribución de %Sac.Caña (Porcentaje de Sacarosa)')
    plt.xlabel('% Sacarosa')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_historico['%Sac.Caña'].dropna(), color='green')
    plt.title('Boxplot de %Sac.Caña para detectar outliers')
    plt.xlabel('% Sacarosa')
    plt.tight_layout()
    plt.show()

    # Paso 5: Creación de categorías de desempeño

    print("--- Creando categorías de desempeño para TCH y %Sac.Caña ---")

    # Definir etiquetas para las categorías
    labels = ['Bajo', 'Medio', 'Alto']

    # Crear categorías para TCH usando terciles (3 grupos de igual tamaño)
    df_historico['Nivel_TCH'] = pd.qcut(df_historico['TCH'], q=3, labels=labels)

    # Crear categorías para %Sac.Caña (manejando valores nulos)
    # Se categorizan solo los valores no nulos
    df_historico['Nivel_Sacarosa'] = pd.qcut(df_historico['%Sac.Caña'][df_historico['%Sac.Caña'].notna()], q=3, labels=labels)

    print("Categorías creadas exitosamente.\n")

    # Mostrar la distribución de las nuevas categorías
    print("--- Distribución de niveles de TCH ---")
    print(df_historico['Nivel_TCH'].value_counts())
    print("\n")
    print("--- Distribución de niveles de %Sac.Caña ---")
    print(df_historico['Nivel_Sacarosa'].value_counts())
    print("\n")

    # Visualizar la distribución de las categorías
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='Nivel_TCH', data=df_historico, palette='viridis', order=labels)
    plt.title('Distribución de Niveles de TCH')

    plt.subplot(1, 2, 2)
    sns.countplot(x='Nivel_Sacarosa', data=df_historico, palette='plasma', order=labels)
    plt.title('Distribución de Niveles de %Sac.Caña')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: No se encontró el archivo 'HISTORICO_SUERTES.xlsx - Hoja1.csv'")


# --- ANÁLISIS DEL DATASET: BD_IPSA_1940.xlsx ---

print("\n" + "="*50)
print("Análisis del archivo: BD_IPSA_1940.xlsx")
print("="*50)

# Paso 6: Cargar y analizar el segundo conjunto de datos
try:
    df_ipsa = pd.read_csv('BD_IPSA_1940.xlsx - BD_IPSA.csv')
    print("Archivo 'BD_IPSA_1940.xlsx - BD_IPSA.csv' cargado exitosamente.")
    print(f"El dataset tiene {df_ipsa.shape[0]} filas y {df_ipsa.shape[1]} columnas.\n")

    # Revisar si hay valores faltantes
    print("--- Revisión de valores faltantes en BD_IPSA ---")
    print(df_ipsa[['TCH', 'sacarosa']].isnull().sum())
    print("El dataset no tiene valores faltantes en las columnas de interés.\n")

    # Crear categorías también para este dataset para la tarea de clasificación
    df_ipsa['Nivel_TCH'] = pd.qcut(df_ipsa['TCH'], q=3, labels=labels)
    df_ipsa['Nivel_Sacarosa'] = pd.qcut(df_ipsa['sacarosa'], q=3, labels=labels)

    print("--- Distribución de niveles de TCH (BD_IPSA) ---")
    print(df_ipsa['Nivel_TCH'].value_counts())
    print("\n")
    print("--- Distribución de niveles de Sacarosa (BD_IPSA) ---")
    print(df_ipsa['Nivel_Sacarosa'].value_counts())
    print("\n")

except FileNotFoundError:
    print("Error: No se encontró el archivo 'BD_IPSA_1940.xlsx - BD_IPSA.csv'")