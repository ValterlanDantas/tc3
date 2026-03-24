# Clustering de aeroportos — exemplo visual

Demonstração didática de **aprendizado não supervisionado** aplicada a dados de voos: agrupa aeroportos por perfil operacional (volume, atrasos, cancelamentos, etc.), reduz dimensões com **PCA** e gera gráficos em alta resolução.

O pipeline está implementado em [`exemplo_clustering_visual.py`](exemplo_clustering_visual.py).

## O que o script faz

1. Carrega `flights.csv` e `airports.csv` e agrega métricas por aeroporto de **origem**.
2. Constrói features: total de voos, atraso médio na partida, taxa de cancelamento, distância média e taxa de desvios (apenas aeroportos com ≥ 100 voos e sem valores faltantes).
3. Normaliza as features com `StandardScaler` (evita que variáveis em escalas diferentes dominem o clustering).
4. Avalia **K** de 2 a 10 com **método do cotovelo** (inércia) e **silhouette score**.
5. Aplica **K-Means** com **K = 4** (fixo no exemplo, após análise visual dos gráficos).
6. Projeta os dados em 2D com **PCA** e plota clusters com centroides.
7. Interpreta cada cluster (rótulos como “hubs eficientes”, “regionais pontuais”, etc.) e compara médias por cluster em gráficos de barras.

## Requisitos

- Python 3.10+ recomendado  
- Dependências: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Instalação:

```bash
pip install -r requirements.txt
```

## Dados necessários

Coloque na **mesma pasta** do script (ou ajuste os caminhos no código):

| Arquivo        | Uso |
|----------------|-----|
| `flights.csv`  | Colunas usadas: `ORIGIN_AIRPORT`, `FLIGHT_NUMBER`, `DEPARTURE_DELAY`, `CANCELLED`, `DISTANCE`, `DIVERTED` |
| `airports.csv` | Carregado no script (base de aeroportos; o foco da agregação é `flights.csv`) |

> Use um dataset de voos compatível com essas colunas (por exemplo, conjuntos públicos de voos nos EUA em formato tabular).

## Como executar

```bash
python exemplo_clustering_visual.py
```

O script imprime cada etapa no terminal e abre janelas do Matplotlib (`plt.show()`). Três imagens PNG são salvas em [`images/`](images/).

### Galeria (para o README no GitHub)

Inclua a pasta `images/` no repositório (faça commit dos `.png` junto com o `README.md`) para as figuras aparecerem na página inicial do projeto no GitHub.

![Método do cotovelo e silhouette por K](images/exemplo_01_elbow_method.png)

![Clusters no plano PCA e distribuição por cluster](images/exemplo_02_clusters_visualizacao.png)

![Comparação das características médias por cluster](images/exemplo_03_caracteristicas_clusters.png)

> **Dica:** rode `python exemplo_clustering_visual.py` uma vez com os CSVs no lugar; depois adicione `git add images/*.png` antes do push.

## Saídas geradas

| Arquivo em `images/` | Conteúdo |
|----------------------|----------|
| `exemplo_01_elbow_method.png` | Cotovelo (inércia) e silhouette por K |
| `exemplo_02_clusters_visualizacao.png` | Clusters no plano PCA + barras de contagem por cluster |
| `exemplo_03_caracteristicas_clusters.png` | Comparação de médias por cluster (voos, atraso, cancelamento, distância) |

## Personalização rápida

- **Número de clusters:** altere a variável `optimal_k` (por padrão `4`) após analisar `exemplo_01_elbow_method.png`.
- **Filtro mínimo de voos:** linha com `TOTAL_VOOS >= 100`.
- **Faixa de K no cotovelo:** `K_range = range(2, 11)`.

## Licença

Defina a licença do repositório conforme sua instituição ou preferência (por exemplo MIT, CC-BY ou uso apenas educacional).

---

*Projeto de exemplo para estudo de machine learning / FIAP — módulo de dados.*
