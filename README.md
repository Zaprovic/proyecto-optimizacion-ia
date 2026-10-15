# Segmentación RFM de clientes con optimización de hiperparámetros

Proyecto de optimización aplicada a IA: segmentar ~640 mil clientes de tarjeta débito
según su comportamiento de compra y **optimizar los hiperparámetros** de los algoritmos
de clustering que generan esa segmentación.

## Problema

Cada cliente se describe con tres variables **RFM** (ventana de 6 meses):

| Variable | Columna | Significado |
|---|---|---|
| Recencia | `recency_dias` | Días desde la última transacción |
| Frecuencia | `frecuencia` | Número de transacciones |
| Monto | `valor_total` | Valor total transado (COP) |

Además se usan dos variables categóricas (`desc_segmento`, `desc_genero`) en K-Prototypes.

## Formulación del problema de optimización

$$\theta^* = \arg\max_{\theta \in \Theta} S(\theta) \quad \text{s.a.} \quad \min_j \frac{n_j(\theta)}{n} \geq 5\\%$$

- **Variables de decisión (θ):** hiperparámetros de cada algoritmo.
- **Función objetivo (S):** coeficiente de silhouette sobre las variables RFM escaladas.
- **Restricción:** ningún cluster con menos del 5 % de los clientes (penalización S = −1).
- **Espacio de búsqueda:** k ≥ 3 (k = 2 solo separa activos de inactivos).

La función objetivo es una caja negra (cada evaluación entrena un modelo), por lo que se
comparan tres estrategias con el mismo presupuesto de evaluaciones:
**Grid Search**, **Random Search** y **optimización bayesiana (TPE, Optuna)**.

| Algoritmo | Hiperparámetros |
|---|---|
| K-Means | `n_clusters`, `init`, `n_init` |
| Gaussian Mixture | `n_components`, `covariance_type`, `reg_covar`, `init_params` |
| K-Prototypes | `n_clusters`, `gamma`, `init` |

Los mejores modelos se comparan en una muestra reservada (silhouette, Davies-Bouldin,
Calinski-Harabasz y estabilidad con ARI), el ganador se reentrena con todos los clientes
y se perfilan los segmentos resultantes.

## Estructura

```
src/
├── data/tabla_total.parquet   # datos de clientes (débito y crédito)
└── main.ipynb                 # análisis completo
```

Los datos no se versionan (`src/data/` está en `.gitignore`): antes de ejecutar el
notebook hay que copiar `tabla_total.parquet` en `src/data/`.

## Cómo ejecutarlo

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.14.

```bash
uv sync
cd src
uv run jupyter nbconvert --to notebook --execute --inplace main.ipynb   # o abrir main.ipynb en VS Code / Jupyter
```

La ejecución completa toma alrededor de 15 minutos (la mayor parte es la búsqueda de hiperparámetros).
