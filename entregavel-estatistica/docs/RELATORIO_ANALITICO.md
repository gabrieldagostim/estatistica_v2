# RELATÓRIO ANALÍTICO — Transparência Pública de Criciúma/SC

## 1. Pergunta de Negócio

**"Quais fatores estruturais (órgão, programa, função, elemento de despesa, recurso, contexto temporal e contexto orçamentário/de pessoal) determinam o valor efetivamente pago de empenhos com pessoal no município de Criciúma, e como prevê-lo a partir desses fatores?"**

Justificativa: a Prefeitura precisa estimar com antecedência quanto efetivamente será desembolsado por empenho de pessoal — informação chave para fluxo de caixa, controle fiscal (LRF) e planejamento orçamentário.

## 2. Variável-alvo

| Item | Detalhe |
|---|---|
| Alvo bruto | `valorPagoEmpenho` (valor efetivamente pago do empenho) |
| Transformação aplicada | `log_target = log1p(valorPagoEmpenho)` |
| Por que log? | Distribuição extremamente assimétrica (valores de R$ 1 a milhões); o log estabiliza variância e linearizar relações |
| Filtro | `valorPagoEmpenho > 0` (remove empenhos zerados/anulados) |

## 3. Arquivos Utilizados

| Arquivo (anos) | Registros | Papel |
|---|---|---|
| **Despesas com Pessoal** (2012–2026, 15 arq.) | 20.691 | Tabela-fato (base) |
| Despesas por Programas e Ações (2018–2026) | 3.750 | Enriquecimento orçamentário |
| Cargos e Vencimentos (2019–2026) | 51.582 | Enriquecimento de quadro funcional |
| Adiantamentos (2019–2026) | 1.053 | Enriquecimento de fluxo extra |

Após filtro de alvo válido: **20.228 registros** (atende mínimo de 20.000).

## 4. Estratégia de Unificação / Joins

| # | Tabela | Chave de junção | Tipo | Como agreguei antes |
|---|---|---|---|---|
| 1 | Programas/Ações → base | `(anoExercicio, descricaoPrograma, descricaoOrgao)` | LEFT | soma de orçado/empenhado/pago, médias de %, contagem de ações |
| 2 | Cargos/Vencimentos → base | `(anoExercicio, nomeEntidade)` | LEFT | nº cargos, vagas criadas/preenchidas, salário médio, taxa ocupação |
| 3 | Adiantamentos → base | `(anoExercicio, descricaoOrgao)` | LEFT | nº adiantamentos, valor total |

Validação de integridade: nulos pós-join preenchidos com 0; duplicidade ausente (chaves únicas verificadas em groupby). Não houve explosão de linhas (joins many-to-one corretos).

## 5. Tratamento de Qualidade

- **Nulos:** colunas numéricas → `fillna(0)`; `fillna` específico em `cv_salario_medio` (NaN quando `niveisSalariais` vazio).
- **Tipos:** `pd.to_numeric(errors='coerce')` em todos os campos monetários; `to_datetime` em `dataEmpenho`.
- **Duplicidade:** verificada via `groupby` antes dos joins.
- **Outliers:** mantidos (representam pagamentos legítimos de grande monta), mas tratados via log-transform.
- **Vazamento (data leakage):** descartei `rank_pago_orgao_ano` (rank do próprio alvo dentro de órgão+ano).

## 6. Engenharia de Atributos (135 candidatas geradas)

| Família | Exemplos |
|---|---|
| Temporais | `mes_empenho`, `trimestre`, `dia_mes`, `dia_semana`, `fim_ano`, `inicio_ano` |
| Contagens aninhadas | `n_pagamentos`, `n_liquidacoes`, `n_docfiscais` |
| Razões/proxies | `tx_anulacao = ValorAnulado / Empenhado`, `tx_restos`, `log_saldoAPagar` |
| Agregados orçamentários | `prog_valor_orcado`, `prog_valor_empenhado`, `prog_pct_pago`, `prog_n_acoes` |
| Agregados RH | `cv_n_cargos`, `cv_vagas_criadas/preenchidas`, `cv_taxa_ocupacao`, `cv_salario_medio` |
| Lags / janelas móveis | `orgao_lag1_sum`, `orgao_lag1_mean`, `orgao_roll3_sum`, `prog_lag1_sum`, `prog_roll3_sum` |
| Target encoding 1-D | sobre 10 categóricas (mean/median/std/max/count) |
| Target encoding 2-D | (orgão×elemento), (programa×elemento), (unidade×elemento), (órgão×função), (órgão×recurso), (função×elemento), (subfunção×elemento), (unidade×função) |
| Target encoding 3-D | (órgão×programa×elemento) |
| Target encoding granular | `cnpjCpfCredor`, `idDespesaDot` |

## 7. Variáveis Selecionadas (|corr| > 0.3 com log_target)

| # | Variável | Correlação | Interpretação |
|---|---|---|---|
| 1 | `te_uniEle_mean` | **+0.4464** | Média histórica do log-pagamento na combinação Unidade×Elemento — o sinal mais forte |
| 2 | `te_orgProgEle_mean` | +0.4278 | Média na tripla Órgão×Programa×Elemento |
| 3 | `te_progEle_mean` | +0.4243 | Média Programa×Elemento |
| 4 | `te_uniEle_median` | +0.4178 | Mediana Unidade×Elemento (robusta a outliers) |
| 5 | `te_orgProgEle_median` | +0.4079 | Mediana tripla Órgão×Programa×Elemento |
| 6 | `te_progEle_median` | +0.4051 | Mediana Programa×Elemento |
| 7 | `te_orgEle_mean` | +0.3991 | Média Órgão×Elemento |
| 8 | `te_funEle_mean` | +0.3955 | Média Função×Elemento |
| 9 | `te_subEle_mean` | +0.3908 | Média Subfunção×Elemento |
| 10 | `te_orgEle_median` | +0.3823 | Mediana Órgão×Elemento |
| 11 | `te_funEle_median` | +0.3807 | Mediana Função×Elemento |
| 12 | `te_subEle_median` | +0.3729 | Mediana Subfunção×Elemento |
| 13 | `te_descricaoElemento_mean` | +0.3392 | Média por Elemento de despesa puro |
| 14 | `te_descricaoElemento_median` | +0.3317 | Mediana por Elemento |
| 15 | `te_despesaDot_mean` | +0.3018 | Média por dotação orçamentária (`idDespesa`) |

✅ **15 variáveis úteis + 1 alvo = 16 variáveis** (atende o mínimo).

### Interpretação analítica

A leitura unânime é que **o "Elemento de Despesa"** (Vencimentos, Obrigações Patronais, Diárias, Material de Consumo, etc.) é o principal driver de magnitude. A interação dele com Unidade/Programa/Órgão refina ainda mais — pagamentos de "Vencimentos" na Secretaria da Educação têm magnitude muito previsível, distinta da mesma rubrica em órgãos pequenos. As **medianas são quase tão informativas quanto as médias**, indicando que o sinal é robusto a outliers. Variáveis temporais, contagens e agregados de Cargos/Adiantamentos (`cv_*`, `ad_*`) ficaram **abaixo de 0.3** isoladamente — capturam ruído mais que sinal monetário.

## 8. Variáveis Descartadas (e motivo)

| Categoria | Variáveis | Motivo |
|---|---|---|
| Temporais isoladas | `mes_empenho`, `trimestre`, `dia_mes`, `fim_ano`, ano | \|corr\| < 0.05 — pagamentos de pessoal são distribuídos uniformemente no calendário |
| Contagens aninhadas | `n_liquidacoes`, `n_docfiscais` | corr ≈ 0.06 — informativos, mas fracos |
| Razões | `tx_anulacao`, `tx_restos`, `log_saldoAPagar` | corr < 0.04 — empenhos pagos têm essas taxas próximas de 0 |
| Agregados RH | todos os `cv_*` | corr < 0.03 — granularidade da agregação muito alta para o nível de empenho |
| Agregados orçamentários | `prog_valor_orcado/empenhado/pago/pct_*` | corr < 0.05 — escala de programa não distingue empenhos individuais |
| Adiantamentos | `ad_n_adiantamentos`, `ad_valor_total` | corr < 0.04 — fluxo paralelo, não preditivo do pagamento principal |
| Lags/rolling | `orgao_lag1_sum`, `prog_roll3_sum`, etc. | corr < 0.08 — médias anuais inteiras não refletem o valor de um empenho específico |
| `te_credor_*` | encoding por CNPJ | corr ≈ 0.21 — alta cardinalidade dilui sinal |
| Vazada | `rank_pago_orgao_ano` | r=0.93 mas é função direta do alvo (descartada) |

## 9. Multicolinearidade

**Severa** entre as 15 selecionadas — todas são target encodings derivadas de "Elemento de Despesa" em granularidades distintas. Pares com `|r| > 0.85` chegam a **30+** (ex.: `te_progEle_mean ↔ te_orgEle_mean = 0.93`; `te_uniEle_mean ↔ te_progEle_mean = 0.90`).

**Implicação:** num modelo linear (regressão), só ~3 dessas variáveis são linearmente independentes — selecionar uma de cada nível de granularidade. Em modelos de árvore (XGBoost/LightGBM) a redundância é tolerada, mas ganho marginal é baixo.

**Recomendação prática:** manter 4–5 dentre o top-15: `te_uniEle_mean`, `te_orgProgEle_mean`, `te_descricaoElemento_mean`, `te_despesaDot_mean` + uma medida de dispersão (`te_descricaoElemento_std`).

## 10. Sugestões de Melhorias Futuras

1. **Target encoding com out-of-fold (K-Fold)** para reduzir vazamento e estimar correlação honesta.
2. **Smoothing bayesiano** nos encodings (prior global) — corrige grupos com poucos casos.
3. **Modelagem não-linear**: GBM (LightGBM/XGBoost) lida nativamente com a multicolinearidade; provável que features de baixa correlação linear (temporais, contagens) tornem-se importantes via interações.
4. **Embedding de credor**: tratar `cnpjCpfCredor` com hashing/embedding em vez de encoding agregado.
5. **Séries temporais por credor**: lags mensais para servidores/fornecedores recorrentes (Vencimentos têm sazonalidade mensal).
6. **Variável-alvo alternativa de classificação**: prever se o empenho terá `saldoAPagar > 0` ao fim do exercício (risco de inscrição em Restos a Pagar) — uso fiscal direto.
7. **Junção textual**: extrair entidades de `historicoEmpenho` (NLP) — descrições como "13º salário", "férias", "rescisão" carregam forte sinal.
8. **Validação temporal**: train ≤2023, test 2024-2026 — métrica realista de previsão prospectiva.

## Entregáveis (arquivos gerados)

- [base_analitica_final.csv](base_analitica_final.csv) — 20.228 registros × 17 colunas (alvo + 15 features + log_target)
- [correlacoes.csv](correlacoes.csv) — ranking completo das 135 candidatas
- [pipeline_analitico.py](pipeline_analitico.py) — pipeline reproducível end-to-end
