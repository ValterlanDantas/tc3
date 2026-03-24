"""
================================================================================
PREPARAÇÃO DE DADOS PARA APRENDIZADO NÃO SUPERVISIONADO
Tech Challenge Fase 3 - FIAP
================================================================================

Este script prepara os dados de voos para algoritmos de aprendizado não 
supervisionado (clustering, redução de dimensionalidade, etc.).

OBJETIVO:
- Selecionar features relevantes para clustering
- Tratar valores ausentes
- Codificar variáveis categóricas
- Normalizar/padronizar features numéricas
- Remover outliers extremos (opcional)
- Criar dataset pronto para ML não supervisionado

ALGORITMOS QUE PODEM SER APLICADOS:
- K-Means Clustering
- DBSCAN
- Hierarchical Clustering
- PCA (Principal Component Analysis)
- t-SNE
- Gaussian Mixture Models

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configurações
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-whitegrid')

print("="*100)
print("PREPARACAO DE DADOS PARA APRENDIZADO NAO SUPERVISIONADO")
print("="*100)

# ================================================================================
# PASSO 1: CARREGAMENTO DOS DADOS
# ================================================================================
print("\n" + "="*100)
print("PASSO 1: CARREGAMENTO DOS DADOS")
print("="*100)

print("\nCarregando datasets...")
airlines_df = pd.read_csv('airlines.csv')
airports_df = pd.read_csv('airports.csv')
flights_df = pd.read_csv('flights.csv', low_memory=False)

print(f"[OK] Dados carregados: {len(flights_df):,} voos")

# ================================================================================
# PASSO 2: SELEÇÃO DE FEATURES RELEVANTES
# ================================================================================
print("\n" + "="*100)
print("PASSO 2: SELECAO DE FEATURES RELEVANTES")
print("="*100)
print("""
EXPLICACAO:
Para aprendizado nao supervisionado, precisamos selecionar features que:
- Sejam relevantes para encontrar padroes
- Nao sejam apenas identificadores (ID, codigos unicos)
- Tenham informacao util para agrupamento
- Nao sejam muito esparsas (muitos valores ausentes)
""")

# Features selecionadas para clustering
features_selecionadas = [
    # Informações temporais
    'MONTH',
    'DAY',
    'DAY_OF_WEEK',
    'SCHEDULED_DEPARTURE',
    'SCHEDULED_ARRIVAL',
    
    # Companhia e aeroportos
    'AIRLINE',
    'ORIGIN_AIRPORT',
    'DESTINATION_AIRPORT',
    
    # Métricas de desempenho
    'DEPARTURE_DELAY',
    'ARRIVAL_DELAY',
    'AIR_TIME',
    'DISTANCE',
    'ELAPSED_TIME',
    
    # Status do voo
    'CANCELLED',
    'DIVERTED',
    
    # Atrasos específicos (quando disponível)
    'WEATHER_DELAY',
    'AIRLINE_DELAY',
    'SECURITY_DELAY',
    'LATE_AIRCRAFT_DELAY',
]

print(f"\nFeatures selecionadas: {len(features_selecionadas)}")
for i, feat in enumerate(features_selecionadas, 1):
    print(f"  {i}. {feat}")

# Criar dataset reduzido
df_ml = flights_df[features_selecionadas].copy()
print(f"\n[OK] Dataset criado: {df_ml.shape[0]:,} linhas x {df_ml.shape[1]} colunas")

# ================================================================================
# PASSO 3: ANÁLISE DE VALORES AUSENTES
# ================================================================================
print("\n" + "="*100)
print("PASSO 3: ANALISE E TRATAMENTO DE VALORES AUSENTES")
print("="*100)
print("""
EXPLICACAO:
Algoritmos de ML nao lidam bem com valores ausentes (NaN).
Estrategias de tratamento:
1. Remover colunas com muitos valores ausentes (>50%)
2. Remover linhas com valores ausentes criticos
3. Imputar valores (preencher com media, mediana, moda)
""")

print("\n3.1. Valores ausentes por coluna:")
print("-" * 100)
missing_info = pd.DataFrame({
    'Coluna': df_ml.columns,
    'Valores_Ausentes': df_ml.isnull().sum(),
    'Percentual': (df_ml.isnull().sum() / len(df_ml) * 100).round(2)
})
missing_info = missing_info[missing_info['Valores_Ausentes'] > 0].sort_values('Valores_Ausentes', ascending=False)
print(missing_info)

# Estratégia 1: Remover colunas com >50% de valores ausentes
print("\n3.2. Removendo colunas muito esparsas (>50% ausentes):")
print("-" * 100)
threshold = 0.5
colunas_para_remover = missing_info[missing_info['Percentual'] > threshold*100]['Coluna'].tolist()
print(f"Colunas a remover: {colunas_para_remover}")
df_ml = df_ml.drop(columns=colunas_para_remover)
print(f"[OK] Dataset apos remocao: {df_ml.shape}")

# Estratégia 2: Remover voos cancelados (para simplificar)
print("\n3.3. Removendo voos cancelados:")
print("-" * 100)
voos_cancelados = df_ml['CANCELLED'].sum()
print(f"Voos cancelados: {voos_cancelados:,} ({voos_cancelados/len(df_ml)*100:.2f}%)")
df_ml = df_ml[df_ml['CANCELLED'] == 0].copy()
print(f"[OK] Dataset apos remocao: {df_ml.shape[0]:,} linhas")

# Estratégia 3: Preencher valores ausentes restantes
print("\n3.4. Imputando valores ausentes restantes:")
print("-" * 100)

# Para variáveis numéricas: preencher com mediana
numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df_ml[col].isnull().sum() > 0:
        mediana = df_ml[col].median()
        df_ml[col].fillna(mediana, inplace=True)
        print(f"  - {col}: preenchido com mediana = {mediana:.2f}")

# Para variáveis categóricas: preencher com moda
categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df_ml[col].isnull().sum() > 0:
        moda = df_ml[col].mode()[0]
        df_ml[col].fillna(moda, inplace=True)
        print(f"  - {col}: preenchido com moda = {moda}")

print(f"\n[OK] Total de valores ausentes restantes: {df_ml.isnull().sum().sum()}")

# Garantir que não há mais NaN (remover linhas com NaN se houver)
if df_ml.isnull().sum().sum() > 0:
    print("\n3.5. Removendo linhas com valores ausentes restantes:")
    print("-" * 100)
    linhas_antes = len(df_ml)
    df_ml = df_ml.dropna()
    linhas_removidas = linhas_antes - len(df_ml)
    print(f"  Linhas removidas: {linhas_removidas:,}")
    print(f"[OK] Dataset final: {df_ml.shape[0]:,} linhas")
    print(f"[OK] Sem valores ausentes: {df_ml.isnull().sum().sum()} NaN")

# ================================================================================
# PASSO 4: ENGENHARIA DE FEATURES
# ================================================================================
print("\n" + "="*100)
print("PASSO 4: ENGENHARIA DE FEATURES")
print("="*100)
print("""
EXPLICACAO:
Criar novas features que podem revelar padroes:
- Features derivadas de tempo
- Categorias de distancia
- Indicadores de atraso
- Agregacoes
""")

# 4.1. Hora do dia (manhã, tarde, noite)
print("\n4.1. Criando categoria de periodo do dia:")
print("-" * 100)
df_ml['HORA_PARTIDA'] = df_ml['SCHEDULED_DEPARTURE'] // 100
df_ml['PERIODO_DIA'] = pd.cut(df_ml['HORA_PARTIDA'], 
                               bins=[0, 6, 12, 18, 24],
                               labels=['Madrugada', 'Manha', 'Tarde', 'Noite'])
print(df_ml['PERIODO_DIA'].value_counts())

# 4.2. Categoria de distância
print("\n4.2. Criando categoria de distancia:")
print("-" * 100)
df_ml['CATEGORIA_DISTANCIA'] = pd.cut(df_ml['DISTANCE'],
                                       bins=[0, 500, 1000, 2000, 5000],
                                       labels=['Curta', 'Media', 'Longa', 'Muito_Longa'])
print(df_ml['CATEGORIA_DISTANCIA'].value_counts())

# 4.3. Indicador de atraso significativo
print("\n4.3. Criando indicador de atraso:")
print("-" * 100)
df_ml['ATRASO_SIGNIFICATIVO'] = (df_ml['DEPARTURE_DELAY'] > 15).astype(int)
print(f"Voos com atraso significativo (>15min): {df_ml['ATRASO_SIGNIFICATIVO'].sum():,}")

# 4.4. Final de semana
print("\n4.4. Criando indicador de final de semana:")
print("-" * 100)
df_ml['FIM_DE_SEMANA'] = df_ml['DAY_OF_WEEK'].isin([6, 7]).astype(int)
print(f"Voos em fim de semana: {df_ml['FIM_DE_SEMANA'].sum():,}")

# 4.5. Eficiência do voo (razão entre tempo real e distância)
print("\n4.5. Criando metrica de eficiencia:")
print("-" * 100)
df_ml['EFICIENCIA_VOO'] = df_ml['AIR_TIME'] / (df_ml['DISTANCE'] + 1)  # +1 para evitar divisao por zero
print(f"Eficiencia media: {df_ml['EFICIENCIA_VOO'].mean():.4f}")

print("\n[OK] Features engenheiradas criadas!")

# ================================================================================
# PASSO 5: CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
# ================================================================================
print("\n" + "="*100)
print("PASSO 5: CODIFICACAO DE VARIAVEIS CATEGORICAS")
print("="*100)
print("""
EXPLICACAO:
Algoritmos de ML trabalham apenas com numeros.
Variaveis categoricas precisam ser convertidas em numeros.

Metodos:
1. Label Encoding: converte categorias em numeros (0, 1, 2, ...)
2. One-Hot Encoding: cria uma coluna binaria para cada categoria
3. Target Encoding: usa estatisticas da variavel alvo

Para clustering, usaremos Label Encoding para evitar muitas colunas.
""")

# Identificar colunas categóricas
categorical_cols = df_ml.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nColunas categoricas encontradas: {len(categorical_cols)}")
for col in categorical_cols:
    print(f"  - {col}: {df_ml[col].nunique()} categorias unicas")

# Label Encoding
print("\n5.1. Aplicando Label Encoding:")
print("-" * 100)
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_ml[f'{col}_ENCODED'] = le.fit_transform(df_ml[col].astype(str))
    le_dict[col] = le
    print(f"  [OK] {col} -> {col}_ENCODED")

# Remover colunas originais categóricas
df_ml = df_ml.drop(columns=categorical_cols)
print(f"\n[OK] Dataset apos encoding: {df_ml.shape}")

# ================================================================================
# PASSO 6: TRATAMENTO DE OUTLIERS
# ================================================================================
print("\n" + "="*100)
print("PASSO 6: TRATAMENTO DE OUTLIERS")
print("="*100)
print("""
EXPLICACAO:
Outliers extremos podem distorcer algoritmos de clustering.
Estrategias:
1. Remover outliers usando IQR (Intervalo Interquartil)
2. Winsorization (limitar valores extremos)
3. Transformacao logaritmica

Vamos remover outliers extremos apenas nas variaveis de atraso.
""")

# Função para detectar outliers usando IQR
def detectar_outliers_iqr(df, coluna, fator=3.0):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - fator * IQR
    limite_superior = Q3 + fator * IQR
    outliers = (df[coluna] < limite_inferior) | (df[coluna] > limite_superior)
    return outliers, limite_inferior, limite_superior

# Detectar e remover outliers extremos em DEPARTURE_DELAY
print("\n6.1. Tratando outliers em DEPARTURE_DELAY:")
print("-" * 100)
outliers_delay, lim_inf, lim_sup = detectar_outliers_iqr(df_ml, 'DEPARTURE_DELAY', fator=3.0)
print(f"Outliers detectados: {outliers_delay.sum():,} ({outliers_delay.sum()/len(df_ml)*100:.2f}%)")
print(f"Limites: [{lim_inf:.2f}, {lim_sup:.2f}]")
df_ml_clean = df_ml[~outliers_delay].copy()
print(f"[OK] Dataset apos remocao: {df_ml_clean.shape[0]:,} linhas")

# ================================================================================
# PASSO 7: NORMALIZAÇÃO/PADRONIZAÇÃO
# ================================================================================
print("\n" + "="*100)
print("PASSO 7: NORMALIZACAO/PADRONIZACAO")
print("="*100)
print("""
EXPLICACAO:
Features em escalas diferentes podem dominar o algoritmo de clustering.
Por exemplo: DISTANCE (0-5000) vs DAY_OF_WEEK (1-7)

Metodos:
1. StandardScaler: media=0, desvio=1 (Z-score normalization)
2. MinMaxScaler: valores entre 0 e 1
3. RobustScaler: resistente a outliers

Para clustering, StandardScaler eh o mais comum.
""")

# Separar features numéricas para normalização
numeric_features = df_ml_clean.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nFeatures numericas para normalizacao: {len(numeric_features)}")

# StandardScaler
print("\n7.1. Aplicando StandardScaler:")
print("-" * 100)
scaler = StandardScaler()
df_scaled = df_ml_clean.copy()
df_scaled[numeric_features] = scaler.fit_transform(df_ml_clean[numeric_features])

print("[OK] Normalizacao concluida!")
print("\nEstatisticas apos normalizacao (primeiras 5 features):")
print(df_scaled[numeric_features[:5]].describe())

# ================================================================================
# PASSO 8: AMOSTRAGEM (OPCIONAL PARA DATASETS GRANDES)
# ================================================================================
print("\n" + "="*100)
print("PASSO 8: AMOSTRAGEM PARA PROTOTIPAGEM")
print("="*100)
print("""
EXPLICACAO:
Com 5+ milhoes de registros, alguns algoritmos podem ser lentos.
Para prototipagem e testes, podemos criar amostras menores.

Estrategias:
1. Amostragem aleatoria simples
2. Amostragem estratificada
3. Amostragem sistematica
""")

# Criar diferentes tamanhos de amostra
sample_sizes = [10000, 50000, 100000]
samples_dict = {}

print("\n8.1. Criando amostras:")
print("-" * 100)
for size in sample_sizes:
    if size <= len(df_scaled):
        sample = df_scaled.sample(n=size, random_state=42)
        samples_dict[f'sample_{size}'] = sample
        print(f"  [OK] Amostra de {size:,} registros criada")

# ================================================================================
# PASSO 9: SALVAR DATASETS PREPARADOS
# ================================================================================
print("\n" + "="*100)
print("PASSO 9: SALVANDO DATASETS PREPARADOS")
print("="*100)

# Dataset completo limpo (sem normalização)
df_ml_clean.to_csv('dados_ml_limpos.csv', index=False)
print(f"[OK] dados_ml_limpos.csv salvo: {df_ml_clean.shape}")

# Dataset completo normalizado
df_scaled.to_csv('dados_ml_normalizados.csv', index=False)
print(f"[OK] dados_ml_normalizados.csv salvo: {df_scaled.shape}")

# Salvar amostras
for name, sample_df in samples_dict.items():
    filename = f'{name}_normalizado.csv'
    sample_df.to_csv(filename, index=False)
    print(f"[OK] {filename} salvo: {sample_df.shape}")

# Salvar scaler para uso futuro
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("[OK] scaler.pkl salvo")

# Salvar label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)
print("[OK] label_encoders.pkl salvo")

# ================================================================================
# PASSO 10: RELATÓRIO FINAL
# ================================================================================
print("\n" + "="*100)
print("RELATORIO FINAL - DADOS PREPARADOS PARA ML NAO SUPERVISIONADO")
print("="*100)

print(f"""
RESUMO DO PROCESSAMENTO:
{"="*100}

1. DADOS ORIGINAIS:
   - Registros: {len(flights_df):,}
   - Features originais: {len(features_selecionadas)}

2. APOS LIMPEZA:
   - Registros: {len(df_ml_clean):,}
   - Features finais: {len(df_ml_clean.columns)}
   - Reducao: {(1 - len(df_ml_clean)/len(flights_df))*100:.2f}%

3. TRANSFORMACOES APLICADAS:
   ✓ Selecao de features relevantes
   ✓ Remocao de valores ausentes
   ✓ Engenharia de features
   ✓ Codificacao de categoricas (Label Encoding)
   ✓ Tratamento de outliers extremos
   ✓ Normalizacao (StandardScaler)
   ✓ Criacao de amostras para prototipagem

4. ARQUIVOS GERADOS:
   ✓ dados_ml_limpos.csv           - Dataset limpo sem normalizacao
   ✓ dados_ml_normalizados.csv     - Dataset completo normalizado
   ✓ sample_10000_normalizado.csv  - Amostra de 10k registros
   ✓ sample_50000_normalizado.csv  - Amostra de 50k registros
   ✓ sample_100000_normalizado.csv - Amostra de 100k registros
   ✓ scaler.pkl                    - Objeto StandardScaler
   ✓ label_encoders.pkl            - Dicionario de LabelEncoders

5. PROXIMOS PASSOS:
   ➜ Aplicar algoritmos de clustering (K-Means, DBSCAN, etc.)
   ➜ Determinar numero otimo de clusters (Elbow Method, Silhouette)
   ➜ Visualizar clusters (PCA, t-SNE)
   ➜ Interpretar e validar os clusters encontrados

{"="*100}

DADOS PRONTOS PARA MACHINE LEARNING! 🚀
""")

print("\n[✓] PROCESSAMENTO CONCLUIDO COM SUCESSO!")
