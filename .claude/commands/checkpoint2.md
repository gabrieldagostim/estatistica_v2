# Checkpoint 2 — Tamanho Amostral + Regressão OLS

Execute o Checkpoint 2 do Trabalho de Extensão de Estatística.

## O que fazer

1. Verificar se `bases/base_ols.csv` existe. Se não existir, executar `scripts/02_features_ols.py` primeiro.
2. Executar `python scripts/04_checkpoint2_amostragem_ols.py` a partir de `c:\Users\eric.1925\Desktop\estatistica_v2`
3. Ler `resultados/checkpoint2_ols_summary.txt` e `resultados/checkpoint2_metricas.json`
4. Interpretar e reportar os resultados detalhadamente

## Interpretação dos Resultados de Amostragem

```
Fórmula: n = z² × p(1-p) / E²
Com: z=1.96 (95% confiança), p=0.5 (máxima variância), E=0.05 (5% erro)
→ n_inicial = 384
Correção finita (N=20.228): n_ajustado = n / (1 + (n-1)/N)
→ n_final ≈ 377 (ou similar)

Conclusão: Temos 20.228 registros >> n_necessário → amostra mais que suficiente
```

## Interpretação do Modelo OLS

Ao ler o `summary()` do statsmodels, reportar:

### Qualidade do Ajuste
- **R²**: proporção da variância explicada
  - < 0.30 → fraco | 0.30–0.50 → aceitável | 0.50–0.70 → bom | > 0.70 → excelente
- **F-statistic e p-valor**: significância global do modelo
  - p < 0.05 → modelo significativo

### Coeficientes
Para cada coeficiente significativo (p < 0.05):
- Sinal: positivo (aumenta pagamento) ou negativo (diminui)
- Magnitude: incremento em log_target por unidade
- IC 95%: intervalo de confiança do coeficiente

### Pressupostos
- **VIF**: todas as features devem ter VIF < 10 (sem multicolinearidade)
- **Normalidade dos resíduos**: Shapiro-Wilk em amostra de resíduos
- **Homocedasticidade**: Breusch-Pagan test (p > 0.05 → homocedasticidade OK)

## Critério de Sucesso

Para o Checkpoint 2 ser completo:
- Tamanho amostral calculado e justificado
- Modelo OLS com pelo menos 3 coeficientes significativos (p < 0.05)
- R² ≥ 0.30
- VIF < 10 para todos os preditores
- Gráficos de diagnóstico salvos em `graficos/checkpoint2_*.png`
- Summary salvo em `resultados/checkpoint2_ols_summary.txt`

## Exemplo de Relatório

```
CHECKPOINT 2 — RESULTADOS
==========================
Tamanho amostral:
  N = 20.228, z = 1.96, p = 0.5, E = 0.05
  n_necessário = 377 | disponível = 20.228 ✅ Suficiente

Modelo OLS (statsmodels):
  Variável dependente: log_target
  R² = 0.72 | F = 1834.5 (p < 0.001) ✅

Coeficientes significativos (p < 0.05):
  log_valorEmpenhado: β = 0.91 (IC: [0.89, 0.93])
  elemento_Vencimentos: β = 0.45 (IC: [0.41, 0.49])
  n_pagamentos: β = 0.18 (IC: [0.15, 0.21])
  ano_empenho: β = 0.02 (IC: [0.01, 0.03])

Pressupostos:
  VIF máximo: 3.2 ✅ (todos < 10)
  Normalidade resíduos: W=0.994, p=0.08 ✅
  Breusch-Pagan: p = 0.12 ✅ Homocedasticidade
```
