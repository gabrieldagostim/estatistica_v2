# Trabalho de Extensao - Estatistica Aplicada

Analise estatistica dos dados de transparencia da Prefeitura Municipal de Criciuma/SC (2012-2026), com foco em Despesas com Pessoal.

## Pergunta-problema

Quais fatores estruturais (elemento de despesa, orgao, valor empenhado, contexto temporal) determinam o valor efetivamente pago nos empenhos de pessoal do municipio de Criciuma/SC, e como prever esse valor?

## Variavel-alvo

- Bruta: `valorPagoEmpenho`
- Modelada: `log_target = log1p(valorPagoEmpenho)`
- Filtro aplicado: `valorPagoEmpenho > 0` (remove empenhos zerados/anulados)

## Estrutura do projeto

```
estatistica_v2/
├── dados/                    # JSONs brutos da Transparencia Criciuma
├── bases/                    # Bases tabulares geradas
├── resultados/               # JSONs e .pkl com metricas, modelo e relatorio
├── graficos/                 # PNGs gerados pelo pipeline
├── docs/                     # Documentacao analitica (Box-Cox, relatorio)
├── pipeline_completo.py      # ARQUIVO UNICO - executa tudo
├── requirements.txt          # Dependencias
└── README.md
```

## Como rodar

1. Criar e ativar um ambiente virtual:

   ```
   python -m venv .venv
   .venv\Scripts\activate         (Windows)
   source .venv/bin/activate      (Linux/Mac)
   ```

2. Instalar dependencias:

   ```
   pip install -r requirements.txt
   ```

3. Executar o pipeline:

   ```
   python pipeline_completo.py
   ```

O script roda end-to-end em poucos segundos e produz todos os artefatos.

## O que o pipeline faz

| Bloco | Etapa | Saida principal |
|-------|-------|-----------------|
| 1 | Carga dos JSONs + feature engineering OLS | `bases/base_ols.csv` |
| 2 | Shapiro-Wilk + Box-Cox + tamanho amostral | `resultados/checkpoint1_resultado.json` |
| 3 | Ajuste OLS (statsmodels) + VIF, Breusch-Pagan, SW residuos | `resultados/checkpoint2_ols_summary.txt`, `checkpoint2_metricas.json`, `checkpoint2_modelo.pkl` |
| 4 | Estimativas pontuais e intervalares (IC media, IC proporcao, IC diferenca) | `resultados/checkpoint3_estimativas.json` |
| 5 | Previsoes OLS para ano futuro com intervalo de predicao 95% | `resultados/checkpoint3_estimativas.json` |
| 6 | Graficos e relatorio final consolidado | `graficos/*.png`, `resultados/relatorio_final.json` |

## Entregaveis cobertos

- Dataset com mais de 20.000 registros (20.228 apos filtro)
- Variavel-alvo definida e justificada
- Teste de normalidade Shapiro-Wilk + transformacao Box-Cox
- Calculo do tamanho amostral com correcao para populacao finita
- Modelo de regressao OLS (statsmodels) com verificacao de pressupostos (VIF, Breusch-Pagan, SW residuos)
- Estimativas pontuais (media, mediana, desvio padrao, erro padrao)
- Estimativas intervalares (IC media log e em R$, IC com FPC, IC para proporcao, IC para diferenca de medias)
- Previsoes OLS para ano futuro com intervalo de predicao de 95%

## Bibliotecas utilizadas (apenas as permitidas pelas aulas)

- pandas, numpy
- scipy (`stats`, `shapiro`, `norm`, `t`, `boxcox`)
- statsmodels (`smf.ols`, `variance_inflation_factor`, `het_breuschpagan`)
- scikit-learn (`train_test_split`, `LabelEncoder`)
- matplotlib, seaborn
- math, json, glob, os, warnings (stdlib)

## Formula do modelo OLS

```
log_target ~ log_valorEmpenhado + n_pagamentos + ano_empenho
           + trim_2 + trim_3 + trim_4
           + C(elemento_cat) + C(tipo_empenho)
```

## Documentacao analitica

- `docs/RELATORIO_ANALITICO.md`: relatorio completo da analise
- `docs/NORMALIZACAO_BOXCOX.md`: detalhes tecnicos da transformacao Box-Cox
- `docs/RESUMO_NORMALIZACAO.md`: resumo executivo da normalizacao
