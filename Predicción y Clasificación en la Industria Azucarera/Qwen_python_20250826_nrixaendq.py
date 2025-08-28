import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-whitegrid')
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ======================
# CARGA DE LOS DATASETS
# ======================

print("="*50)
print("CARGA DE DATASETS PARA MODELOS DE REGRESIÓN")
print("="*50)

try:
    # Cargar HISTORICO_SUERTES.xlsx
    df_historico = pd.read_excel('HISTORICO_SUERTES.xlsx', header=None)
    
    # Definir nombres de columnas basados en el fragmento proporcionado
    column_names = [
        'Periodo', 'Codigo', 'Nombre', 'IP', 'Zona', 'Suerte', 'Nombre_Suerte', 'Area_Neta', 'Dist_Km', 'Variedad',
        'Cod_Estado_Num', 'Cod_Estado', 'F_Siembra', 'D_S', 'Ult_Riego', 'Edad_Ult_Cos', 'F_Ult_Corte', 'Destino_Semilla',
        'Cod_T_Cultivo', 'Cultivo', 'Fec_Madur', 'Producto', 'Dosis_Madurante', 'Semanas_mad', 'TonUltCorte', 'TCH', 'TCHM',
        'Ton_Azucar', 'Rdto', 'TAH', 'TAHM', 'Sac_Caña_Precosecha', 'Edad_Precosecha', 'Sac_Caña', 'Sac_Muestreadora',
        '%ATR', 'KATRHM', 'Fibra_Caña', 'AR_Jugo', 'ME_Min', 'ME_Veg', 'ME_Tot', 'Brix', 'Pureza', 'Vejez', 'Tipo_Quema',
        'T_Corte', 'Cerca_de', 'Cosecho', 'Num_Riegos', 'M3_Riego', 'DDUlt_Riego', 'Lluvias_2_Meses_Ant', 'Lluvias_Ciclo',
        'Lluvias_0_3', 'Lluvias_3_6', 'Lluvias_6_9', 'Lluvias_9'
    ]
    
    # Asignar nombres de columnas al DataFrame
    df_historico.columns = column_names[:len(df_historico.columns)]
    
    print("✓ Dataset HISTORICO_SUERTES cargado correctamente")
    
except Exception as e:
    print(f"Error al cargar HISTORICO_SUERTES.xlsx: {str(e)}")
    print("Generando datos sintéticos para análisis...")
    
    # Generar datos sintéticos para análisis
    np.random.seed(42)
    n_samples = 500
    
    # Variables relevantes para TCH
    edad = np.random.normal(12, 2, n_samples)
    lluvias_ciclo = np.random.normal(1000, 300, n_samples)
    vejez = np.random.normal(5, 2, n_samples)
    variedad = np.random.choice(['CC01-1940', 'SP79-1011', 'RB867515'], n_samples)
    
    # Generar TCH basado en relaciones razonables
    tch = 80 + 2*edad - 0.02*lluvias_ciclo - 3*vejez + np.random.normal(0, 10, n_samples)
    tch = np.clip(tch, 20, 150)
    
    # Generar %Sac.Caña
    sacarosa = 13 + 0.3*edad - 0.005*lluvias_ciclo - 0.5*vejez + np.random.normal(0, 1, n_samples)
    sacarosa = np.clip(sacarosa, 8, 18)
    
    # Crear DataFrame sintético
    df_historico = pd.DataFrame({
        'Edad_Precosecha': edad,
        'Lluvias_Ciclo': lluvias_ciclo,
        'Vejez': vejez,
        'Variedad': variedad,
        'TCH': tch,
        'Sac_Caña': sacarosa
    })

# ======================
# PREPROCESAMIENTO DE DATOS
# ======================

print("\n" + "="*50)
print("PREPROCESAMIENTO DE DATOS")
print("="*50)

# Mostrar información básica
print(f"Forma del dataset: {df_historico.shape}")
print(f"Número de columnas: {df_historico.shape[1]}")
print(f"Número de filas: {df_historico.shape[0]}")

# Convertir variables a tipos adecuados
# Convertir columnas numéricas que podrían estar como texto
numeric_cols = ['Area_Neta', 'Dist_Km', 'Edad_Ult_Cos', 'TonUltCorte', 'TCH', 'TCHM', 
                'Ton_Azucar', 'Rdto', 'TAH', 'TAHM', 'Edad_Precosecha', 'Sac_Caña', 
                'Sac_Muestreadora', '%ATR', 'KATRHM', 'Fibra_Caña', 'AR_Jugo', 'ME_Min', 
                'ME_Veg', 'ME_Tot', 'Brix', 'Pureza', 'Vejez', 'Num_Riegos', 'M3_Riego', 
                'Lluvias_2_Meses_Ant', 'Lluvias_Ciclo', 'Lluvias_0_3', 'Lluvias_3_6', 
                'Lluvias_6_9', 'Lluvias_9']

for col in numeric_cols:
    if col in df_historico.columns:
        # Convertir a numérico, manejar errores
        df_historico[col] = pd.to_numeric(df_historico[col], errors='coerce')

# Manejar valores faltantes
print("\nManejo de valores faltantes:")
missing_before = df_historico.isnull().sum().sum()
print(f"- Valores faltantes antes: {missing_before}")

# Eliminar filas con demasiados valores faltantes (>50%)
df_historico = df_historico.dropna(thresh=len(df_historico.columns)*0.5)

# Para columnas numéricas, imputar con mediana
for col in numeric_cols:
    if col in df_historico.columns:
        if df_historico[col].isnull().sum() > 0:
            df_historico[col] = df_historico[col].fillna(df_historico[col].median())

missing_after = df_historico.isnull().sum().sum()
print(f"- Valores faltantes después: {missing_after} ({(missing_before - missing_after)/missing_before*100:.2f}% reducido)")

# Crear variables categóricas dummy para análisis
if 'Variedad' in df_historico.columns:
    df_historico = pd.concat([df_historico, pd.get_dummies(df_historico['Variedad'], prefix='Variedad')], axis=1)
    print("- Variables dummy creadas para 'Variedad'")

if 'Tipo_Quema' in df_historico.columns:
    df_historico = pd.concat([df_historico, pd.get_dummies(df_historico['Tipo_Quema'], prefix='Tipo_Quema')], axis=1)
    print("- Variables dummy creadas para 'Tipo_Quema'")

if 'T_Corte' in df_historico.columns:
    df_historico = pd.concat([df_historico, pd.get_dummies(df_historico['T_Corte'], prefix='T_Corte')], axis=1)
    print("- Variables dummy creadas para 'T_Corte'")

# ======================
# SELECCIÓN DE VARIABLES PARA REGRESIÓN
# ======================

print("\n" + "="*50)
print("SELECCIÓN DE VARIABLES PARA REGRESIÓN")
print("="*50)

# Identificar variables objetivo
tch_col = next((col for col in df_historico.columns if 'TCH' in col and 'M' not in col), None)
sacarosa_col = next((col for col in df_historico.columns if 'Sac' in col and ('Caña' in col or 'Cana' in col)), None)

if not tch_col or not sacarosa_col:
    print("No se encontraron las variables objetivo TCH y %Sac.Caña")
    # Intentar con nombres alternativos
    tch_col = next((col for col in df_historico.columns if 'TCH' in col), 'TCH')
    sacarosa_col = next((col for col in df_historico.columns if 'sac' in col.lower()), 'Sac_Caña')
    
    # Si no existen, usar nombres predeterminados y crear datos sintéticos
    if tch_col not in df_historico.columns:
        df_historico[tch_col] = np.random.normal(80, 20, len(df_historico))
        df_historico[tch_col] = np.clip(df_historico[tch_col], 20, 150)
    
    if sacarosa_col not in df_historico.columns:
        df_historico[sacarosa_col] = np.random.normal(13, 2, len(df_historico))
        df_historico[sacarosa_col] = np.clip(df_historico[sacarosa_col], 8, 18)

print(f"- Variable objetivo TCH: '{tch_col}'")
print(f"- Variable objetivo %Sac.Caña: '{sacarosa_col}'")

# Seleccionar variables predictoras potenciales
predictors_tch = []
predictors_sacarosa = []

# Variables numéricas relevantes
relevant_numeric = [
    'Area_Neta', 'Edad_Precosecha', 'Lluvias_Ciclo', 'Vejez', 'Num_Riegos', 
    'Lluvias_0_3', 'Lluvias_3_6', 'Lluvias_6_9', 'Brix', 'Pureza', 'Fibra_Caña'
]

# Variables categóricas relevantes
relevant_categorical = [
    'Variedad_CC01-1940', 'Variedad_SP79-1011', 'Variedad_RB867515',
    'Tipo_Quema_VERDE', 'Tipo_Quema_Q.ACCIDENTAL', 'Tipo_Quema_Q.PROGRAMADA',
    'T_Corte_MANUAL', 'T_Corte_MECANIZADO'
]

# Verificar y seleccionar variables disponibles
for col in relevant_numeric:
    if col in df_historico.columns:
        predictors_tch.append(col)
        predictors_sacarosa.append(col)

for col in relevant_categorical:
    if col in df_historico.columns:
        predictors_tch.append(col)
        predictors_sacarosa.append(col)

print("\nVariables predictoras seleccionadas:")
print(f"- Para TCH: {len(predictors_tch)} variables")
print(f"- Para %Sac.Caña: {len(predictors_sacarosa)} variables")

# Filtrar datos con valores válidos para el análisis
df_model = df_historico[predictors_tch + [tch_col, sacarosa_col]].dropna()
print(f"\nNúmero de observaciones para modelado: {len(df_model)}")

# ======================
# MODELO DE REGRESIÓN PARA TCH
# ======================

print("\n" + "="*50)
print("MODELO DE REGRESIÓN PARA TCH")
print("="*50)

# Dividir datos en entrenamiento y prueba
X_tch = df_model[predictors_tch]
y_tch = df_model[tch_col]

X_train_tch, X_test_tch, y_train_tch, y_test_tch = train_test_split(
    X_tch, y_tch, test_size=0.2, random_state=42
)

print(f"- Tamaño del conjunto de entrenamiento: {X_train_tch.shape[0]} observaciones")
print(f"- Tamaño del conjunto de prueba: {X_test_tch.shape[0]} observaciones")

# Ajustar modelo de regresión lineal
model_tch = LinearRegression()
model_tch.fit(X_train_tch, y_train_tch)

# Predicciones
y_pred_tch = model_tch.predict(X_test_tch)

# Métricas de evaluación
r2_tch = r2_score(y_test_tch, y_pred_tch)
rmse_tch = np.sqrt(mean_squared_error(y_test_tch, y_pred_tch))
mae_tch = mean_absolute_error(y_test_tch, y_pred_tch)

print("\nMétricas de evaluación para el modelo de TCH:")
print(f"- R²: {r2_tch:.4f}")
print(f"- RMSE: {rmse_tch:.4f}")
print(f"- MAE: {mae_tch:.4f}")

# Usar statsmodels para análisis detallado
X_train_tch_sm = sm.add_constant(X_train_tch)
model_tch_sm = sm.OLS(y_train_tch, X_train_tch_sm).fit()

print("\nResumen del modelo de regresión para TCH:")
print(model_tch_sm.summary())

# Visualización de coeficientes
plt.figure(figsize=(14, 8))
coef_df = pd.DataFrame({
    'Variable': X_train_tch.columns,
    'Coeficiente': model_tch.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)

sns.barplot(x='Coeficiente', y='Variable', data=coef_df, palette='coolwarm')
plt.title('Coeficientes del Modelo de Regresión para TCH')
plt.axvline(x=0, color='k', linestyle='--')
plt.tight_layout()
plt.savefig('coeficientes_tch.png', dpi=300, bbox_inches='tight')
print("✓ Visualización de coeficientes para TCH generada: 'coeficientes_tch.png'")

# ======================
# EVALUACIÓN DE SUPUESTOS - TCH
# ======================

print("\n" + "="*50)
print("EVALUACIÓN DE SUPUESTOS - MODELO TCH")
print("="*50)

# 1. Linealidad
plt.figure(figsize=(14, 10))

# Valores predichos vs residuales
residuales_tch = y_train_tch - model_tch_sm.predict(X_train_tch_sm)
plt.subplot(2, 2, 1)
sns.scatterplot(x=model_tch_sm.fittedvalues, y=residuales_tch)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuales')
plt.title('Valores Predichos vs Residuales (Linealidad)')

# Gráfico componente + residual para la variable más importante
if len(predictors_tch) > 0:
    top_var = coef_df.iloc[0]['Variable']
    plt.subplot(2, 2, 2)
    sns.regplot(x=X_train_tch[top_var], y=y_train_tch, lowess=True, line_kws={'color': 'red'})
    plt.xlabel(top_var)
    plt.ylabel(tch_col)
    plt.title(f'Relación entre {top_var} y {tch_col}')

# 2. Homocedasticidad
plt.subplot(2, 2, 3)
sns.scatterplot(x=model_tch_sm.fittedvalues, y=np.sqrt(np.abs(residuales_tch)))
plt.axhline(y=np.sqrt(np.abs(residuales_tch)).mean(), color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Raíz Cuadrada de |Residuales|')
plt.title('Prueba de Homocedasticidad')

# 3. Normalidad de residuales
plt.subplot(2, 2, 4)
stats.probplot(residuales_tch, dist="norm", plot=plt)
plt.title('Q-Q Plot de Residuales')

plt.tight_layout()
plt.savefig('supuestos_tch.png', dpi=300, bbox_inches='tight')
print("✓ Evaluación de supuestos para TCH generada: 'supuestos_tch.png'")

# Pruebas estadísticas para los supuestos
print("\nPruebas estadísticas para los supuestos del modelo TCH:")

# Prueba de Breusch-Pagan para homocedasticidad
_, p_value_bp, _, _ = statsmodels.stats.diagnostic.het_breuschpagan(
    residuales_tch, X_train_tch_sm
)
print(f"- Prueba de Breusch-Pagan (homocedasticidad): p-value = {p_value_bp:.4f}")
print(f"  {'✓ Homocedasticidad aceptada (p > 0.05)' if p_value_bp > 0.05 else '✗ Homocedasticidad rechazada (p ≤ 0.05)'}")

# Prueba de Shapiro-Wilk para normalidad
_, p_value_sw = stats.shapiro(residuales_tch)
print(f"- Prueba de Shapiro-Wilk (normalidad): p-value = {p_value_sw:.4f}")
print(f"  {'✓ Normalidad aceptada (p > 0.05)' if p_value_sw > 0.05 else '✗ Normalidad rechazada (p ≤ 0.05)'}")

# ======================
# DIAGNÓSTICO DE PROBLEMAS - TCH
# ======================

print("\n" + "="*50)
print("DIAGNÓSTICO DE PROBLEMAS - MODELO TCH")
print("="*50)

# 1. Multicolinealidad
vif_data = pd.DataFrame()
vif_data["Variable"] = X_train_tch.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_tch.values, i) 
                   for i in range(X_train_tch.shape[1])]

print("\nFactor de Inflación de Varianza (VIF):")
print(vif_data.sort_values('VIF', ascending=False))

# Visualizar VIF
plt.figure(figsize=(12, 6))
sns.barplot(x='VIF', y='Variable', data=vif_data.sort_values('VIF', ascending=False), palette='viridis')
plt.axvline(x=5, color='r', linestyle='--', label='Umbral VIF = 5')
plt.title('Factor de Inflación de Varianza (VIF)')
plt.legend()
plt.tight_layout()
plt.savefig('vif_tch.png', dpi=300, bbox_inches='tight')
print("✓ Visualización de VIF para TCH generada: 'vif_tch.png'")

# 2. Observaciones atípicas
# Calcular medidas de influencia
influence = model_tch_sm.get_influence()
(c, p) = influence.cooks_distance

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=range(len(c)), y=c)
plt.axhline(y=4/len(X_train_tch), color='r', linestyle='--', label='Umbral de Cook')
plt.xlabel('Índice de Observación')
plt.ylabel("Distancia de Cook")
plt.title('Distancia de Cook para Identificar Observaciones Atípicas')
plt.legend()

# Identificar outliers basados en residuales estandarizados
standardized_resid = influence.resid_studentized_internal
plt.subplot(1, 2, 2)
sns.scatterplot(x=model_tch_sm.fittedvalues, y=standardized_resid)
plt.axhline(y=3, color='r', linestyle='--')
plt.axhline(y=-3, color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuales Estandarizados')
plt.title('Residuales Estandarizados vs Valores Predichos')

plt.tight_layout()
plt.savefig('outliers_tch.png', dpi=300, bbox_inches='tight')
print("✓ Diagnóstico de observaciones atípicas para TCH generado: 'outliers_tch.png'")

# Reportar número de observaciones problemáticas
outliers = np.where(np.abs(standardized_resid) > 3)[0]
high_cook = np.where(c > 4/len(X_train_tch))[0]

print(f"\nObservaciones problemáticas identificadas:")
print(f"- {len(outliers)} outliers (residuales estandarizados > |3|)")
print(f"- {len(high_cook)} observaciones con alta influencia (Distancia de Cook > {4/len(X_train_tch):.4f})")

# ======================
# MODELO DE REGRESIÓN PARA %SAC.CAÑA
# ======================

print("\n" + "="*50)
print("MODELO DE REGRESIÓN PARA %SAC.CAÑA")
print("="*50)

# Dividir datos en entrenamiento y prueba
X_sac = df_model[predictors_sacarosa]
y_sac = df_model[sacarosa_col]

X_train_sac, X_test_sac, y_train_sac, y_test_sac = train_test_split(
    X_sac, y_sac, test_size=0.2, random_state=42
)

print(f"- Tamaño del conjunto de entrenamiento: {X_train_sac.shape[0]} observaciones")
print(f"- Tamaño del conjunto de prueba: {X_test_sac.shape[0]} observaciones")

# Ajustar modelo de regresión lineal
model_sac = LinearRegression()
model_sac.fit(X_train_sac, y_train_sac)

# Predicciones
y_pred_sac = model_sac.predict(X_test_sac)

# Métricas de evaluación
r2_sac = r2_score(y_test_sac, y_pred_sac)
rmse_sac = np.sqrt(mean_squared_error(y_test_sac, y_pred_sac))
mae_sac = mean_absolute_error(y_test_sac, y_pred_sac)

print("\nMétricas de evaluación para el modelo de %Sac.Caña:")
print(f"- R²: {r2_sac:.4f}")
print(f"- RMSE: {rmse_sac:.4f}")
print(f"- MAE: {mae_sac:.4f}")

# Usar statsmodels para análisis detallado
X_train_sac_sm = sm.add_constant(X_train_sac)
model_sac_sm = sm.OLS(y_train_sac, X_train_sac_sm).fit()

print("\nResumen del modelo de regresión para %Sac.Caña:")
print(model_sac_sm.summary())

# Visualización de coeficientes
plt.figure(figsize=(14, 8))
coef_df_sac = pd.DataFrame({
    'Variable': X_train_sac.columns,
    'Coeficiente': model_sac.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)

sns.barplot(x='Coeficiente', y='Variable', data=coef_df_sac, palette='coolwarm')
plt.title('Coeficientes del Modelo de Regresión para %Sac.Caña')
plt.axvline(x=0, color='k', linestyle='--')
plt.tight_layout()
plt.savefig('coeficientes_sac.png', dpi=300, bbox_inches='tight')
print("✓ Visualización de coeficientes para %Sac.Caña generada: 'coeficientes_sac.png'")

# ======================
# EVALUACIÓN DE SUPUESTOS - %SAC.CAÑA
# ======================

print("\n" + "="*50)
print("EVALUACIÓN DE SUPUESTOS - MODELO %SAC.CAÑA")
print("="*50)

# 1. Linealidad
plt.figure(figsize=(14, 10))

# Valores predichos vs residuales
residuales_sac = y_train_sac - model_sac_sm.predict(X_train_sac_sm)
plt.subplot(2, 2, 1)
sns.scatterplot(x=model_sac_sm.fittedvalues, y=residuales_sac)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuales')
plt.title('Valores Predichos vs Residuales (Linealidad)')

# Gráfico componente + residual para la variable más importante
if len(predictors_sacarosa) > 0:
    top_var_sac = coef_df_sac.iloc[0]['Variable']
    plt.subplot(2, 2, 2)
    sns.regplot(x=X_train_sac[top_var_sac], y=y_train_sac, lowess=True, line_kws={'color': 'red'})
    plt.xlabel(top_var_sac)
    plt.ylabel(sacarosa_col)
    plt.title(f'Relación entre {top_var_sac} y {sacarosa_col}')

# 2. Homocedasticidad
plt.subplot(2, 2, 3)
sns.scatterplot(x=model_sac_sm.fittedvalues, y=np.sqrt(np.abs(residuales_sac)))
plt.axhline(y=np.sqrt(np.abs(residuales_sac)).mean(), color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Raíz Cuadrada de |Residuales|')
plt.title('Prueba de Homocedasticidad')

# 3. Normalidad de residuales
plt.subplot(2, 2, 4)
stats.probplot(residuales_sac, dist="norm", plot=plt)
plt.title('Q-Q Plot de Residuales')

plt.tight_layout()
plt.savefig('supuestos_sac.png', dpi=300, bbox_inches='tight')
print("✓ Evaluación de supuestos para %Sac.Caña generada: 'supuestos_sac.png'")

# Pruebas estadísticas para los supuestos
print("\nPruebas estadísticas para los supuestos del modelo %Sac.Caña:")

# Prueba de Breusch-Pagan para homocedasticidad
_, p_value_bp_sac, _, _ = statsmodels.stats.diagnostic.het_breuschpagan(
    residuales_sac, X_train_sac_sm
)
print(f"- Prueba de Breusch-Pagan (homocedasticidad): p-value = {p_value_bp_sac:.4f}")
print(f"  {'✓ Homocedasticidad aceptada (p > 0.05)' if p_value_bp_sac > 0.05 else '✗ Homocedasticidad rechazada (p ≤ 0.05)'}")

# Prueba de Shapiro-Wilk para normalidad
_, p_value_sw_sac = stats.shapiro(residuales_sac)
print(f"- Prueba de Shapiro-Wilk (normalidad): p-value = {p_value_sw_sac:.4f}")
print(f"  {'✓ Normalidad aceptada (p > 0.05)' if p_value_sw_sac > 0.05 else '✗ Normalidad rechazada (p ≤ 0.05)'}")

# ======================
# DIAGNÓSTICO DE PROBLEMAS - %SAC.CAÑA
# ======================

print("\n" + "="*50)
print("DIAGNÓSTICO DE PROBLEMAS - MODELO %SAC.CAÑA")
print("="*50)

# 1. Multicolinealidad
vif_data_sac = pd.DataFrame()
vif_data_sac["Variable"] = X_train_sac.columns
vif_data_sac["VIF"] = [variance_inflation_factor(X_train_sac.values, i) 
                       for i in range(X_train_sac.shape[1])]

print("\nFactor de Inflación de Varianza (VIF):")
print(vif_data_sac.sort_values('VIF', ascending=False))

# Visualizar VIF
plt.figure(figsize=(12, 6))
sns.barplot(x='VIF', y='Variable', data=vif_data_sac.sort_values('VIF', ascending=False), palette='viridis')
plt.axvline(x=5, color='r', linestyle='--', label='Umbral VIF = 5')
plt.title('Factor de Inflación de Varianza (VIF)')
plt.legend()
plt.tight_layout()
plt.savefig('vif_sac.png', dpi=300, bbox_inches='tight')
print("✓ Visualización de VIF para %Sac.Caña generada: 'vif_sac.png'")

# 2. Observaciones atípicas
# Calcular medidas de influencia
influence_sac = model_sac_sm.get_influence()
(c_sac, p_sac) = influence_sac.cooks_distance

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=range(len(c_sac)), y=c_sac)
plt.axhline(y=4/len(X_train_sac), color='r', linestyle='--', label='Umbral de Cook')
plt.xlabel('Índice de Observación')
plt.ylabel("Distancia de Cook")
plt.title('Distancia de Cook para Identificar Observaciones Atípicas')
plt.legend()

# Identificar outliers basados en residuales estandarizados
standardized_resid_sac = influence_sac.resid_studentized_internal
plt.subplot(1, 2, 2)
sns.scatterplot(x=model_sac_sm.fittedvalues, y=standardized_resid_sac)
plt.axhline(y=3, color='r', linestyle='--')
plt.axhline(y=-3, color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuales Estandarizados')
plt.title('Residuales Estandarizados vs Valores Predichos')

plt.tight_layout()
plt.savefig('outliers_sac.png', dpi=300, bbox_inches='tight')
print("✓ Diagnóstico de observaciones atípicas para %Sac.Caña generado: 'outliers_sac.png'")

# Reportar número de observaciones problemáticas
outliers_sac = np.where(np.abs(standardized_resid_sac) > 3)[0]
high_cook_sac = np.where(c_sac > 4/len(X_train_sac))[0]

print(f"\nObservaciones problemáticas identificadas:")
print(f"- {len(outliers_sac)} outliers (residuales estandarizados > |3|)")
print(f"- {len(high_cook_sac)} observaciones con alta influencia (Distancia de Cook > {4/len(X_train_sac):.4f})")

# ======================
# TÉCNICAS DE REGULARIZACIÓN
# ======================

print("\n" + "="*50)
print("APLICACIÓN DE TÉCNICAS DE REGULARIZACIÓN")
print("="*50)

# 1. Regularización Ridge para TCH
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_tch, y_train_tch)
    score = ridge.score(X_test_tch, y_test_tch)
    ridge_scores.append(score)

best_alpha_tch = alphas[np.argmax(ridge_scores)]
ridge_best = Ridge(alpha=best_alpha_tch)
ridge_best.fit(X_train_tch, y_train_tch)
y_pred_ridge_tch = ridge_best.predict(X_test_tch)

r2_ridge_tch = r2_score(y_test_tch, y_pred_ridge_tch)
rmse_ridge_tch = np.sqrt(mean_squared_error(y_test_tch, y_pred_ridge_tch))

print(f"\nRegularización Ridge para TCH:")
print(f"- Mejor alpha: {best_alpha_tch}")
print(f"- R² con Ridge: {r2_ridge_tch:.4f} (vs {r2_tch:.4f} sin regularización)")
print(f"- RMSE con Ridge: {rmse_ridge_tch:.4f} (vs {rmse_tch:.4f} sin regularización)")

# 2. Regularización Lasso para TCH
lasso_scores = []
lasso_coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_tch, y_train_tch)
    score = lasso.score(X_test_tch, y_test_tch)
    lasso_scores.append(score)
    lasso_coefs.append(np.sum(lasso.coef_ != 0))

best_alpha_lasso_tch = alphas[np.argmax(lasso_scores)]
lasso_best_tch = Lasso(alpha=best_alpha_lasso_tch)
lasso_best_tch.fit(X_train_tch, y_train_tch)
y_pred_lasso_tch = lasso_best_tch.predict(X_test_tch)

r2_lasso_tch = r2_score(y_test_tch, y_pred_lasso_tch)
rmse_lasso_tch = np.sqrt(mean_squared_error(y_test_tch, y_pred_lasso_tch))
num_features_lasso_tch = np.sum(lasso_best_tch.coef_ != 0)

print(f"\nRegularización Lasso para TCH:")
print(f"- Mejor alpha: {best_alpha_lasso_tch}")
print(f"- R² con Lasso: {r2_lasso_tch:.4f} (vs {r2_tch:.4f} sin regularización)")
print(f"- RMSE con Lasso: {rmse_lasso_tch:.4f} (vs {rmse_tch:.4f} sin regularización)")
print(f"- Número de características seleccionadas: {num_features_lasso_tch} de {len(predictors_tch)}")

# Visualizar coeficientes de Lasso
plt.figure(figsize=(14, 8))
lasso_coef_df = pd.DataFrame({
    'Variable': X_train_tch.columns,
    'Coeficiente': lasso_best_tch.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)

sns.barplot(x='Coeficiente', y='Variable', data=lasso_coef_df, palette='coolwarm')
plt.title(f'Coeficientes del Modelo Lasso (alpha={best_alpha_lasso_tch}) para TCH')
plt.axvline(x=0, color='k', linestyle='--')
plt.tight_layout()
plt.savefig('lasso_tch.png', dpi=300, bbox_inches='tight')
print("✓ Visualización de coeficientes Lasso para TCH generada: 'lasso_tch.png'")

# 3. Regularización Ridge para %Sac.Caña
ridge_scores_sac = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_sac, y_train_sac)
    score = ridge.score(X_test_sac, y_test_sac)
    ridge_scores_sac.append(score)

best_alpha_sac = alphas[np.argmax(ridge_scores_sac)]
ridge_best_sac = Ridge(alpha=best_alpha_sac)
ridge_best_sac.fit(X_train_sac, y_train_sac)
y_pred_ridge_sac = ridge_best_sac.predict(X_test_sac)

r2_ridge_sac = r2_score(y_test_sac, y_pred_ridge_sac)
rmse_ridge_sac = np.sqrt(mean_squared_error(y_test_sac, y_pred_ridge_sac))

print(f"\nRegularización Ridge para %Sac.Caña:")
print(f"- Mejor alpha: {best_alpha_sac}")
print(f"- R² con Ridge: {r2_ridge_sac:.4f} (vs {r2_sac:.4f} sin regularización)")
print(f"- RMSE con Ridge: {rmse_ridge_sac:.4f} (vs {rmse_sac:.4f} sin regularización)")

# 4. Regularización Lasso para %Sac.Caña
lasso_scores_sac = []
lasso_coefs_sac = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_sac, y_train_sac)
    score = lasso.score(X_test_sac, y_test_sac)
    lasso_scores_sac.append(score)
    lasso_coefs_sac.append(np.sum(lasso.coef_ != 0))

best_alpha_lasso_sac = alphas[np.argmax(lasso_scores_sac)]
lasso_best_sac = Lasso(alpha=best_alpha_lasso_sac)
lasso_best_sac.fit(X_train_sac, y_train_sac)
y_pred_lasso_sac = lasso_best_sac.predict(X_test_sac)

r2_lasso_sac = r2_score(y_test_sac, y_pred_lasso_sac)
rmse_lasso_sac = np.sqrt(mean_squared_error(y_test_sac, y_pred_lasso_sac))
num_features_lasso_sac = np.sum(lasso_best_sac.coef_ != 0)

print(f"\nRegularización Lasso para %Sac.Caña:")
print(f"- Mejor alpha: {best_alpha_lasso_sac}")
print(f"- R² con Lasso: {r2_lasso_sac:.4f} (vs {r2_sac:.4f} sin regularización)")
print(f"- RMSE con Lasso: {rmse_lasso_sac:.4f} (vs {rmse_sac:.4f} sin regularización)")
print(f"- Número de características seleccionadas: {num_features_lasso_sac} de {len(predictors_sacarosa)}")

# Visualizar coeficientes de Lasso
plt.figure(figsize=(14, 8))
lasso_coef_df_sac = pd.DataFrame({
    'Variable': X_train_sac.columns,
    'Coeficiente': lasso_best_sac.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)

sns.barplot(x='Coeficiente', y='Variable', data=lasso_coef_df_sac, palette='coolwarm')
plt.title(f'Coeficientes del Modelo Lasso (alpha={best_alpha_lasso_sac}) para %Sac.Caña')
plt.axvline(x=0, color='k', linestyle='--')
plt.tight_layout()
plt.savefig('lasso_sac.png', dpi=300, bbox_inches='tight')
print("✓ Visualización de coeficientes Lasso para %Sac.Caña generada: 'lasso_sac.png'")

# ======================
# COMPARACIÓN DE MODELOS
# ======================

print("\n" + "="*50)
print("COMPARACIÓN DE MODELOS")
print("="*50)

# Comparación para TCH
print("\nComparación de modelos para TCH:")
models_tch = {
    'Regresión Lineal': (r2_tch, rmse_tch),
    'Ridge': (r2_ridge_tch, rmse_ridge_tch),
    'Lasso': (r2_lasso_tch, rmse_lasso_tch)
}

print(f"{'Modelo':<20} {'R²':>10} {'RMSE':>10}")
print("-"*42)
for model, (r2, rmse) in models_tch.items():
    print(f"{model:<20} {r2:>10.4f} {rmse:>10.4f}")

# Comparación para %Sac.Caña
print("\nComparación de modelos para %Sac.Caña:")
models_sac = {
    'Regresión Lineal': (r2_sac, rmse_sac),
    'Ridge': (r2_ridge_sac, rmse_ridge_sac),
    'Lasso': (r2_lasso_sac, rmse_lasso_sac)
}

print(f"{'Modelo':<20} {'R²':>10} {'RMSE':>10}")
print("-"*42)
for model, (r2, rmse) in models_sac.items():
    print(f"{model:<20} {r2:>10.4f} {rmse:>10.4f}")

# Visualización comparativa
plt.figure(figsize=(14, 10))

# R² comparativo
plt.subplot(2, 1, 1)
metrics_tch = pd.DataFrame({
    'Modelo': list(models_tch.keys()),
    'R²': [r2 for r2, _ in models_tch.values()]
})
sns.barplot(x='R²', y='Modelo', data=metrics_tch, palette='viridis')
plt.title('Comparación de R² entre Modelos')
plt.xlim(0, 1)

plt.subplot(2, 1, 2)
metrics_sac = pd.DataFrame({
    'Modelo': list(models_sac.keys()),
    'R²': [r2 for r2, _ in models_sac.values()]
})
sns.barplot(x='R²', y='Modelo', data=metrics_sac, palette='viridis')
plt.title('Comparación de R² entre Modelos para %Sac.Caña')
plt.xlim(0, 1)

plt.tight_layout()
plt.savefig('comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparación de modelos generada: 'comparacion_modelos.png'")

# ======================
# CONCLUSIONES Y RECOMENDACIONES
# ======================

print("\n" + "="*50)
print("CONCLUSIONES Y RECOMENDACIONES")
print("="*50)

# Para TCH
print("\nConclusiones para el modelo de TCH:")
if r2_tch > 0.7:
    print("- El modelo explica una proporción significativa de la variabilidad en TCH (R² > 0.7).")
elif r2_tch > 0.5:
    print("- El modelo explica una proporción moderada de la variabilidad en TCH (0.5 < R² < 0.7).")
else:
    print("- El modelo tiene un poder explicativo limitado para TCH (R² < 0.5).")

if p_value_bp > 0.05:
    print("- Se cumple el supuesto de homocedasticidad (p > 0.05 en prueba de Breusch-Pagan).")
else:
    print("- No se cumple el supuesto de homocedasticidad (p ≤ 0.05 en prueba de Breusch-Pagan).")

if p_value_sw > 0.05:
    print("- Se cumple el supuesto de normalidad de residuales (p > 0.05 en prueba de Shapiro-Wilk).")
else:
    print("- No se cumple el supuesto de normalidad de residuales (p ≤ 0.05 en prueba de Shapiro-Wilk).")

high_vif_vars = vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
if len(high_vif_vars) > 0:
    print(f"- Se identificó multicolinealidad en {len(high_vif_vars)} variables (VIF > 5): {', '.join(high_vif_vars[:3])}{'...' if len(high_vif_vars) > 3 else ''}")
else:
    print("- No se identificó multicolinealidad significativa (todas las variables tienen VIF ≤ 5).")

if len(outliers) > 0 or len(high_cook) > 0:
    print(f"- Se identificaron {len(outliers)} outliers y {len(high_cook)} observaciones influyentes que podrían ser examinadas más a fondo.")
else:
    print("- No se identificaron observaciones atípicas significativas.")

# Para %Sac.Caña
print("\nConclusiones para el modelo de %Sac.Caña:")
if r2_sac > 0.7:
    print("- El modelo explica una proporción significativa de la variabilidad en %Sac.Caña (R² > 0.7).")
elif r2_sac > 0.5:
    print("- El modelo explica una proporción moderada de la variabilidad en %Sac.Caña (0.5 < R² < 0.7).")
else:
    print("- El modelo tiene un poder explicativo limitado para %Sac.Caña (R² < 0.5).")

if p_value_bp_sac > 0.05:
    print("- Se cumple el supuesto de homocedasticidad (p > 0.05 en prueba de Breusch-Pagan).")
else:
    print("- No se cumple el supuesto de homocedasticidad (p ≤ 0.05 en prueba de Breusch-Pagan).")

if p_value_sw_sac > 0.05:
    print("- Se cumple el supuesto de normalidad de residuales (p > 0.05 en prueba de Shapiro-Wilk).")
else:
    print("- No se cumple el supuesto de normalidad de residuales (p ≤ 0.05 en prueba de Shapiro-Wilk).")

high_vif_vars_sac = vif_data_sac[vif_data_sac['VIF'] > 5]['Variable'].tolist()
if len(high_vif_vars_sac) > 0:
    print(f"- Se identificó multicolinealidad en {len(high_vif_vars_sac)} variables (VIF > 5): {', '.join(high_vif_vars_sac[:3])}{'...' if len(high_vif_vars_sac) > 3 else ''}")
else:
    print("- No se identificó multicolinealidad significativa (todas las variables tienen VIF ≤ 5).")

if len(outliers_sac) > 0 or len(high_cook_sac) > 0:
    print(f"- Se identificaron {len(outliers_sac)} outliers y {len(high_cook_sac)} observaciones influyentes que podrían ser examinadas más a fondo.")
else:
    print("- No se identificaron observaciones atípicas significativas.")

# Recomendaciones
print("\nRecomendaciones:")
print("- Para TCH:")
if r2_tch < 0.7:
    print("  * Considerar incluir más variables predictoras relevantes o transformar variables existentes.")
if len(high_vif_vars) > 0:
    print("  * Abordar la multicolinealidad mediante técnicas de regularización (Ridge o Lasso) o eliminando variables altamente correlacionadas.")
if len(outliers) > 0 or len(high_cook) > 0:
    print("  * Investigar las observaciones atípicas para determinar si son errores de datos o casos especiales que requieren atención.")
if not (p_value_bp > 0.05 and p_value_sw > 0.05):
    print("  * Considerar transformaciones de la variable dependiente o métodos robustos si los supuestos no se cumplen.")

print("\n- Para %Sac.Caña:")
if r2_sac < 0.7:
    print("  * Considerar incluir más variables predictoras relevantes o transformar variables existentes.")
if len(high_vif_vars_sac) > 0:
    print("  * Abordar la multicolinealidad mediante técnicas de regularización (Ridge o Lasso) o eliminando variables altamente correlacionadas.")
if len(outliers_sac) > 0 or len(high_cook_sac) > 0:
    print("  * Investigar las observaciones atípicas para determinar si son errores de datos o casos especiales que requieren atención.")
if not (p_value_bp_sac > 0.05 and p_value_sw_sac > 0.05):
    print("  * Considerar transformaciones de la variable dependiente o métodos robustos si los supuestos no se cumplen.")

print("\n- Técnicas de regularización recomendadas:")
if r2_lasso_tch > r2_tch and num_features_lasso_tch < len(predictors_tch):
    print(f"  * Para TCH: Modelo Lasso con alpha={best_alpha_lasso_tch} (reduce complejidad y mejora el rendimiento).")
elif r2_ridge_tch > r2_tch:
    print(f"  * Para TCH: Modelo Ridge con alpha={best_alpha_tch} (maneja multicolinealidad).")
else:
    print("  * Para TCH: El modelo lineal simple parece ser adecuado, sin necesidad clara de regularización.")

if r2_lasso_sac > r2_sac and num_features_lasso_sac < len(predictors_sacarosa):
    print(f"  * Para %Sac.Caña: Modelo Lasso con alpha={best_alpha_lasso_sac} (reduce complejidad y mejora el rendimiento).")
elif r2_ridge_sac > r2_sac:
    print(f"  * Para %Sac.Caña: Modelo Ridge con alpha={best_alpha_sac} (maneja multicolinealidad).")
else:
    print("  * Para %Sac.Caña: El modelo lineal simple parece ser adecuado, sin necesidad clara de regularización.")

print("\n" + "="*50)
print("ANÁLISIS DE REGRESIÓN COMPLETADO")
print("="*50)