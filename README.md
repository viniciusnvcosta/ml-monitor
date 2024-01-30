pyscora-monitor
===
## passos da amostra (Minerador)
---

### 1. dados de entrada

- pegar nome do modelo

> `s3://oncase-sebraepe/dev/staging/models/recsys_crossvalidation_model_produto_pf/recsys_crossvalidation_model_produto_pf.pkl`

- `recsys_crossvalidation_model_produto_pf`

- adicionar hash da execução

- pegar y_test(true) no bucket de silver

- `.parquet`

- pegar y_pred(inferência) no bucket de gold

### 2. script de run do tracker

- definir como puxar os ids projeto e tenant

- definir como criar o tracker correto(`TrackRegression`, `TrackClassification`, `TrackClustering`)

- definir **perfumaria** do mlflow (nome do projeto, tenant, timestamp)

- rodar as **métricas**


### 3. script de run do detector

- pegar do app (mlflow) as novas métricas criadas
- comparar com a métrica anterior
- analisar a credencidade do resultado (advergência do resultado)
- notificar caso existir advergência