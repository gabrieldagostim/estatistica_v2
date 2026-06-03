# Checkpoint 1 — Normalidade e Box-Cox

Execute o Checkpoint 1 do Trabalho de Extensão de Estatística.

## O que fazer

1. Verificar se `bases/base_ols.csv` existe. Se não existir, executar `scripts/02_features_ols.py` primeiro.
2. Executar `python scripts/03_checkpoint1_normalidade.py` a partir do diretório `c:\Users\eric.1925\Desktop\estatistica_v2`
3. Ler o arquivo `resultados/checkpoint1_resultado.json` gerado
4. Interpretar e reportar os resultados

## Interpretação dos Resultados

Após executar o script, reportar:

- **Número de registros**: deve ser ≥ 20.000 ✅
- **Variável-alvo**: `log_target = log1p(valorPagoEmpenho)`
- **Shapiro-Wilk (original)**: W e p-valor
  - Se p < 0.05 → distribuição NÃO normal → Box-Cox necessário
  - Se p ≥ 0.05 → distribuição normal → Box-Cox opcional
- **Box-Cox λ**: valor do parâmetro ótimo
- **Shapiro-Wilk (após Box-Cox)**: W e p-valor — deve ser mais alto que o original
- **Melhoria de normalidade**: percentual de melhoria no p-valor

## Critério de Sucesso

Para o Checkpoint 1 ser considerado completo:
- ≥ 25 variáveis candidatas identificadas
- ≥ 15 com |correlação| > 0.3 com o target  
- Teste Shapiro-Wilk executado e documentado
- Box-Cox aplicado com λ reportado
- Gráficos salvos em `graficos/checkpoint1_*.png`

## Exemplo de Relatório

```
CHECKPOINT 1 — RESULTADOS
==========================
Dataset: 20.228 registros válidos (2012–2026)
Variável-alvo: log_target = log1p(valorPagoEmpenho)
Variáveis candidatas: 28 identificadas, 15+ com |r| > 0.3

Teste Shapiro-Wilk (original):
  W = 0.9931, p = 7.19e-15 → NÃO normal (p < 0.05)

Transformação Box-Cox:
  λ ótimo = 1.2624
  
Teste Shapiro-Wilk (após Box-Cox):
  W = 0.9966, p = 4.00e-04
  Melhoria: 17.98× no p-valor
  Redução de assimetria: 81.8%

Gráficos: graficos/checkpoint1_histogramas.png, graficos/checkpoint1_qq.png
```
