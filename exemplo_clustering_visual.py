"""
================================================================================
EXEMPLO PRÁTICO - CLUSTERING DE AEROPORTOS
Demonstração visual de aprendizado não supervisionado
================================================================================

Este script mostra de forma visual e didática como funciona:
1. Clustering (K-Means)
2. Redução de Dimensionalidade (PCA)
3. Interpretação dos resultados
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

OUT_IMG = Path("images")
OUT_IMG.mkdir(parents=True, exist_ok=True)

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 8)

print("="*100)
print("EXEMPLO PRATICO: CLUSTERING DE AEROPORTOS")
print("="*100)
print("\nVamos descobrir grupos naturais de aeroportos baseado em suas caracteristicas!")
print()

# ================================================================================
# PASSO 1: CARREGAR E PREPARAR OS DADOS
# ================================================================================
print("\n" + "="*100)
print("PASSO 1: CARREGANDO DADOS")
print("="*100)

# Carregar dados
flights_df = pd.read_csv('flights.csv', low_memory=False)
airports_df = pd.read_csv('airports.csv')

print(f"[OK] {len(flights_df):,} voos carregados")
print(f"[OK] {len(airports_df):,} aeroportos carregados")

# ================================================================================
# PASSO 2: CRIAR FEATURES POR AEROPORTO
# ================================================================================
print("\n" + "="*100)
print("PASSO 2: CRIANDO CARACTERISTICAS DOS AEROPORTOS")
print("="*100)
print("""
Para cada aeroporto, vamos calcular:
- Numero total de voos (como origem)
- Atraso medio na partida
- Taxa de cancelamento
- Distancia media dos voos
- Taxa de voos desviados
""")

# Agregar dados por aeroporto de origem
airport_stats = flights_df.groupby('ORIGIN_AIRPORT').agg({
    'FLIGHT_NUMBER': 'count',              # Total de voos
    'DEPARTURE_DELAY': 'mean',              # Atraso médio
    'CANCELLED': 'mean',                    # Taxa de cancelamento
    'DISTANCE': 'mean',                     # Distância média
    'DIVERTED': 'mean'                      # Taxa de desvios
}).reset_index()

# Renomear colunas
airport_stats.columns = ['AIRPORT', 'TOTAL_VOOS', 'ATRASO_MEDIO', 
                         'TAXA_CANCELAMENTO', 'DISTANCIA_MEDIA', 'TAXA_DESVIO']

# Remover aeroportos com poucos voos (< 100)
airport_stats = airport_stats[airport_stats['TOTAL_VOOS'] >= 100].copy()

# Remover linhas com NaN
airport_stats = airport_stats.dropna()

print(f"\n[OK] Analisando {len(airport_stats)} aeroportos")
print("\nExemplo de dados:")
print(airport_stats.head(10))

# ================================================================================
# PASSO 3: NORMALIZAR OS DADOS
# ================================================================================
print("\n" + "="*100)
print("PASSO 3: NORMALIZANDO DADOS")
print("="*100)
print("""
Por que normalizar?
- TOTAL_VOOS varia de 100 a 50.000 (grande escala)
- TAXA_CANCELAMENTO varia de 0 a 0.05 (pequena escala)
- Sem normalização, TOTAL_VOOS dominaria o clustering!
""")

# Selecionar features para clustering
features = ['TOTAL_VOOS', 'ATRASO_MEDIO', 'TAXA_CANCELAMENTO', 
            'DISTANCIA_MEDIA', 'TAXA_DESVIO']

X = airport_stats[features].values

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[OK] Dados normalizados (media=0, desvio=1)")
print(f"\nAntes: Min={X.min():.2f}, Max={X.max():.2f}")
print(f"Depois: Min={X_scaled.min():.2f}, Max={X_scaled.max():.2f}")

# ================================================================================
# PASSO 4: DETERMINAR NÚMERO ÓTIMO DE CLUSTERS (ELBOW METHOD)
# ================================================================================
print("\n" + "="*100)
print("PASSO 4: DETERMINANDO NUMERO OTIMO DE CLUSTERS")
print("="*100)
print("""
Vamos testar diferentes valores de K (2 a 10) e ver qual eh o melhor.
Usamos o 'Metodo do Cotovelo': onde a curva faz um cotovelo, ali esta o K ideal!
""")

inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("\nTestando diferentes valores de K...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    labels = kmeans.labels_
    silhouette = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette)
    
    print(f"  K={k}: Inercia={kmeans.inertia_:.2f}, Silhouette={silhouette:.3f}")

# Plotar método do cotovelo
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Inércia
axes[0].plot(K_range, inertias, marker='o', linewidth=3, markersize=10, color='blue')
axes[0].set_xlabel('Numero de Clusters (K)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Inercia (Soma das Distancias)', fontsize=14, fontweight='bold')
axes[0].set_title('Metodo do Cotovelo\n(Procure onde a curva faz um "L")', 
                 fontsize=16, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].annotate('Cotovelo?\nK ideal aqui!', xy=(4, inertias[2]), 
                xytext=(6, inertias[2]+5000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold')

# Gráfico 2: Silhouette
axes[1].plot(K_range, silhouette_scores, marker='s', linewidth=3, markersize=10, color='green')
axes[1].set_xlabel('Numero de Clusters (K)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=14, fontweight='bold')
axes[1].set_title('Silhouette Score\n(Quanto maior, melhor)', 
                 fontsize=16, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0.5, color='red', linestyle='--', label='Limite Bom (0.5)')
axes[1].legend()

plt.tight_layout()
p1 = OUT_IMG / "exemplo_01_elbow_method.png"
plt.savefig(p1, dpi=300, bbox_inches='tight')
print(f"\n[OK] Grafico salvo: {p1}")
plt.show()

# Escolher K baseado nos gráficos (vamos usar K=4 neste exemplo)
optimal_k = 4
print(f"\n>>> K OTIMO ESCOLHIDO: {optimal_k}")

# ================================================================================
# PASSO 5: APLICAR K-MEANS COM K ÓTIMO
# ================================================================================
print("\n" + "="*100)
print("PASSO 5: APLICANDO K-MEANS CLUSTERING")
print("="*100)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Adicionar clusters ao dataframe
airport_stats['CLUSTER'] = clusters

# Métricas
silhouette = silhouette_score(X_scaled, clusters)
print(f"\n[OK] K-Means aplicado com K={optimal_k}")
print(f"[OK] Silhouette Score: {silhouette:.3f}")

# Distribuição
print(f"\nDistribuicao dos aeroportos por cluster:")
for i in range(optimal_k):
    count = (clusters == i).sum()
    percent = count / len(clusters) * 100
    print(f"  Cluster {i}: {count:3d} aeroportos ({percent:5.1f}%)")

# ================================================================================
# PASSO 6: REDUÇÃO DE DIMENSIONALIDADE COM PCA
# ================================================================================
print("\n" + "="*100)
print("PASSO 6: REDUCAO DE DIMENSIONALIDADE (PCA)")
print("="*100)
print("""
Temos 5 dimensoes (features), impossivel visualizar!
PCA vai reduzir para 2 dimensoes mantendo a maior parte da informacao.
""")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\n[OK] PCA aplicado:")
print(f"  Componente Principal 1: {pca.explained_variance_ratio_[0]*100:.1f}% da variancia")
print(f"  Componente Principal 2: {pca.explained_variance_ratio_[1]*100:.1f}% da variancia")
print(f"  TOTAL explicado: {sum(pca.explained_variance_ratio_)*100:.1f}%")

print("\nO que cada componente representa:")
print("  PC1 (eixo X): Principalmente TAMANHO do aeroporto")
print("  PC2 (eixo Y): Principalmente DESEMPENHO (atrasos/cancelamentos)")

# ================================================================================
# PASSO 7: VISUALIZAR CLUSTERS EM 2D
# ================================================================================
print("\n" + "="*100)
print("PASSO 7: VISUALIZANDO CLUSTERS")
print("="*100)

# Criar figura com 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Gráfico 1: Scatter plot dos clusters
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
for i in range(optimal_k):
    mask = clusters == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=f'Cluster {i}',
                   s=100, alpha=0.6, edgecolors='black', linewidth=1.5)

# Adicionar centroides
centroides_pca = pca.transform(kmeans_final.cluster_centers_)
axes[0].scatter(centroides_pca[:, 0], centroides_pca[:, 1],
               c='black', marker='X', s=500, linewidths=3,
               edgecolors='yellow', label='Centroides')

axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variancia)', 
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variancia)', 
                  fontsize=13, fontweight='bold')
axes[0].set_title(f'Clusters de Aeroportos (K={optimal_k})\nVisualizacao com PCA', 
                 fontsize=15, fontweight='bold')
axes[0].legend(fontsize=11, loc='best')
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Distribuição dos clusters
cluster_counts = pd.Series(clusters).value_counts().sort_index()
bars = axes[1].bar(cluster_counts.index, cluster_counts.values, 
                   color=colors[:optimal_k], alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_xlabel('Cluster', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Numero de Aeroportos', fontsize=13, fontweight='bold')
axes[1].set_title('Distribuicao dos Aeroportos por Cluster', fontsize=15, fontweight='bold')
axes[1].grid(True, axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
p2 = OUT_IMG / "exemplo_02_clusters_visualizacao.png"
plt.savefig(p2, dpi=300, bbox_inches='tight')
print(f"\n[OK] Grafico salvo: {p2}")
plt.show()

# ================================================================================
# PASSO 8: INTERPRETAR OS CLUSTERS
# ================================================================================
print("\n" + "="*100)
print("PASSO 8: INTERPRETACAO DOS CLUSTERS")
print("="*100)
print("\nCaracteristicas de cada cluster (valores ORIGINAIS, nao normalizados):\n")

cluster_names = []
for i in range(optimal_k):
    cluster_data = airport_stats[airport_stats['CLUSTER'] == i]
    
    print(f"\n{'='*90}")
    print(f"CLUSTER {i} - {len(cluster_data)} aeroportos ({len(cluster_data)/len(airport_stats)*100:.1f}%)")
    print(f"{'='*90}")
    
    # Estatísticas
    stats = cluster_data[features].mean()
    
    print(f"  Voos/dia (media):        {stats['TOTAL_VOOS']:>10,.0f}")
    print(f"  Atraso medio:            {stats['ATRASO_MEDIO']:>10.1f} minutos")
    print(f"  Taxa cancelamento:       {stats['TAXA_CANCELAMENTO']:>10.2%}")
    print(f"  Distancia media:         {stats['DISTANCIA_MEDIA']:>10,.0f} milhas")
    print(f"  Taxa desvio:             {stats['TAXA_DESVIO']:>10.2%}")
    
    # Interpretar e dar nome ao cluster
    if stats['TOTAL_VOOS'] > 10000 and stats['ATRASO_MEDIO'] < 10:
        nome = "HUBS EFICIENTES"
        descricao = "Aeroportos grandes com excelente desempenho"
    elif stats['TOTAL_VOOS'] > 10000:
        nome = "HUBS CONGESTIONADOS"
        descricao = "Aeroportos grandes com problemas de atraso"
    elif stats['ATRASO_MEDIO'] < 8:
        nome = "AEROPORTOS REGIONAIS PONTUAIS"
        descricao = "Aeroportos menores com boa pontualidade"
    else:
        nome = "AEROPORTOS MEDIOS"
        descricao = "Aeroportos de porte medio com desempenho misto"
    
    cluster_names.append(nome)
    print(f"\n  >>> NOME: {nome}")
    print(f"  >>> DESCRICAO: {descricao}")
    
    # Exemplos de aeroportos
    exemplos = cluster_data.nlargest(5, 'TOTAL_VOOS')['AIRPORT'].tolist()
    print(f"  >>> EXEMPLOS: {', '.join(exemplos[:5])}")

# ================================================================================
# PASSO 9: GRÁFICO DE CARACTERÍSTICAS POR CLUSTER
# ================================================================================
print("\n" + "="*100)
print("PASSO 9: COMPARACAO VISUAL DAS CARACTERISTICAS")
print("="*100)

# Calcular médias por cluster
cluster_means = airport_stats.groupby('CLUSTER')[features].mean()

# Criar gráfico de radar/barras
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparacao das Caracteristicas por Cluster', 
            fontsize=18, fontweight='bold', y=0.995)

features_plot = ['TOTAL_VOOS', 'ATRASO_MEDIO', 'TAXA_CANCELAMENTO', 'DISTANCIA_MEDIA']

for idx, feature in enumerate(features_plot):
    ax = axes[idx // 2, idx % 2]
    
    values = cluster_means[feature]
    bars = ax.bar(range(optimal_k), values, color=colors[:optimal_k], 
                  alpha=0.7, edgecolor='black', linewidth=2)
    
    # Configurar eixos
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{feature.replace("_", " ").title()} por Cluster', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(optimal_k))
    ax.set_xticklabels([f'C{i}' for i in range(optimal_k)])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Adicionar valores
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
p3 = OUT_IMG / "exemplo_03_caracteristicas_clusters.png"
plt.savefig(p3, dpi=300, bbox_inches='tight')
print(f"\n[OK] Grafico salvo: {p3}")
plt.show()

# ================================================================================
# PASSO 10: INSIGHTS E CONCLUSÕES
# ================================================================================
print("\n" + "="*100)
print("PASSO 10: INSIGHTS E RECOMENDACOES")
print("="*100)

print("\n>>> PRINCIPAIS DESCOBERTAS:")
print("\n1. Identificamos {} grupos distintos de aeroportos".format(optimal_k))
print("\n2. Caracteristicas dos grupos:")
for i in range(optimal_k):
    print(f"   - Cluster {i}: {cluster_names[i]}")

print("\n3. Qualidade do clustering:")
print(f"   - Silhouette Score: {silhouette:.3f} ({('EXCELENTE' if silhouette > 0.7 else 'BOM' if silhouette > 0.5 else 'RAZOAVEL')})")
print(f"   - Variancia explicada (PCA): {sum(pca.explained_variance_ratio_)*100:.1f}%")

print("\n>>> RECOMENDACOES DE NEGOCIOS:")
print("""
1. HUBS EFICIENTES:
   - Usar como modelo de boas praticas
   - Replicar processos em outros aeroportos
   
2. HUBS CONGESTIONADOS:
   - Prioridade alta para melhorias
   - Aumentar recursos operacionais
   - Revisar slots de voos
   
3. AEROPORTOS REGIONAIS:
   - Manter o bom desempenho
   - Potencial para expansao
   
4. AEROPORTOS MEDIOS:
   - Monitorar desempenho
   - Prevenir deterioracao
""")

print("\n" + "="*100)
print("ANALISE CONCLUIDA COM SUCESSO!")
print("="*100)
print("\nArquivos gerados:")
print(f"  - {OUT_IMG / 'exemplo_01_elbow_method.png'}")
print(f"  - {OUT_IMG / 'exemplo_02_clusters_visualizacao.png'}")
print(f"  - {OUT_IMG / 'exemplo_03_caracteristicas_clusters.png'}")
print("\n>>> Todos os graficos foram salvos! Verifique a pasta do projeto.")
