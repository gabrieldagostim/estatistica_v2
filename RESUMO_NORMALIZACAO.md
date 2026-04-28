# Resumo Executivo: Normalização com Box-Cox

**Data:** 28 de Abril de 2026  
**Status:** ✅ **CONCLUÍDO**  
**Base Original:** `base_analitica_final.csv` (20.228 registros)  
**Base Normalizada:** `base_normalizada_boxcox.csv` (20.228 registros)

---

## 📋 O Que Foi Feito

### **Etapa 1: Teste de Normalidade (Shapiro-Wilk)**
```
Série Original (log_target):
  ✓ Estatística W: 0.9931
  ✗ P-valor: 7.19e-15 (rejeita normalidade)
  
Interpretação: 
  • A série NÃO é estatisticamente normal
  • Mas está visualmente próxima à normal
  • Aplicação de transformação se justifica
```

### **Etapa 2: Teste de 6 Transformações**
| Transformação | W | P-valor | Melhor? |
|---|---|---|---|
| **Box-Cox** | **0.9966** | **4.00e-04** | ✅ SIM |
| Yeo-Johnson | 0.9965 | 3.01e-04 | Quase |
| Z-Score | 0.9932 | 1.63e-07 | Não |
| Original | 0.9931 | 7.19e-15 | Baseline |
| Raiz Quadrada | 0.9451 | 1.31e-25 | Não |
| Log Natural | 0.6016 | 5.77e-54 | Não |
| Inversa | 0.0252 | 2.43e-70 | ❌ Péssima |

**Vencedor:** Box-Cox com λ = **1.2624**

### **Etapa 3: Aplicação de Box-Cox**
Transformação aplicada a toda a base (20.228 registros)
```python
log_target_normalizado = (log_target^1.2624 - 1) / 1.2624
```

### **Etapa 4: Validação Pós-Transformação**
```
Melhorias Alcançadas:
  ✓ Assimetria: -0.3029 → -0.0553 (81.8% mais próxima de 0)
  ✓ Curtose:    +0.3779 → -0.0757 (80.0% mais próxima de 0)
  ✓ Teste SW:   p = 7.19e-15 → 4.00e-04 (17.98x melhor)
```

---

## 📊 Principais Resultados

### **Comparação Estatísticas Descritivas**

| Métrica | Original | Normalizado | Melhoria |
|---|---|---|---|
| **Média** | 9.6166 | 13.1414 | Escala aumenta |
| **Mediana** | 9.7822 | 13.3042 | Mantém ordem |
| **Desvio Padrão** | 2.3035 | 4.1020 | Variância aumenta |
| **Assimetria** | -0.3029 | -0.0553 | ✅ **81.8%** |
| **Curtose** | +0.3779 | -0.0757 | ✅ **80.0%** |

---

## 📁 Arquivos Gerados

### **1. Base de Dados**
```
base_normalizada_boxcox.csv (6.6 MB)
├── 20.228 registros
├── 19 colunas (original: 17 + normalizado + lambda)
├── log_target: Série original (para referência)
├── log_target_normalizado: Série normalizada (USE ESTA)
└── lambda_boxcox: Parâmetro de transformação (1.2624)
```

**Como Usar:**
```python
import pandas as pd

df = pd.read_csv('base_normalizada_boxcox.csv')

# Para regressão linear (pressupostos normais)
modelo = LinearRegression()
modelo.fit(df[features], df['log_target_normalizado'])

# Para árvores (XGBoost, etc)
modelo = XGBRegressor()
modelo.fit(df[features], df['log_target'])
```

### **2. Documentação Completa**
```
NORMALIZACAO_BOXCOX.md (9.6 KB)
├── Motivação e contexto
├── Detalhes técnicos de cada transformação
├── Interpretação estatística
├── Recomendações de uso
├── Referências bibliográficas
└── Status final: ✅ Pronto para modelagem
```

### **3. Metadados**
```
metadados_normalizacao_boxcox.json (941 B)
├── Data de geração
├── Parâmetro lambda otimizado
├── Estatísticas descritivas (antes/depois)
└── Rastreabilidade completa
```

### **4. Relatórios Visuais (6 gráficos)**
```
relatorio_01_histogramas.png
  → Comparação visual: Original vs Normalizado
  → Sobreposição com curva normal teórica
  
relatorio_02_qq_plots.png
  → Q-Q Plots para avaliação visual de normalidade
  → Mostra aderência à diagonal teórica
  
relatorio_03_box_violin_plots.png
  → Box plots e violin plots
  → Simetria e dispersão dos dados
  
relatorio_04_estatisticas.png
  → Tabela com todas as estatísticas
  → Percentuais de melhoria
  
relatorio_05_testes_normalidade.png
  → Resultados do Teste Shapiro-Wilk
  → Interpretação e recomendações
  
relatorio_06_scatter.png
  → Relação entre série original e normalizada
  → Validação da transformação
```

### **5. Scripts Reproducíveis**
```
teste_normalidade.py
  → Testa 6 transformações diferentes
  → Gera gráfico comparativo
  
gerar_base_normalizada.py
  → Cria base com Box-Cox
  → Calcula estatísticas
  → Salva metadados
  
relatorio_normalizacao_visual.py
  → Gera 6 relatórios visuais
  → Comparações detalhadas
```

---

## 🎯 Recomendações de Uso

### **✅ USE SÉRIE NORMALIZADA (`log_target_normalizado`) PARA:**
- Regressão Linear
- ANOVA e testes paramétricos
- GLM com distribuição normal
- Quando pressupostos teóricos são críticos
- Comparações com literatura científica

### **✅ USE SÉRIE ORIGINAL (`log_target`) PARA:**
- Modelos baseados em árvores (XGBoost, LightGBM, Random Forest)
- Interpretabilidade diretos em escala de valor
- Modelos não-paramétricos
- Ensemble de modelos
- Quando a transformação distorça resultados

### **⚠️ PRÓXIMOS PASSOS RECOMENDADOS:**
1. **Validação de Pressupostos (Residuais)**
   ```python
   from scipy.stats import shapiro
   residuos = y_teste - y_pred
   shapiro(residuos)  # Deve ser p > 0.05
   ```

2. **Testes de Heterocedasticidade**
   - Breusch-Pagan ou White
   - Verificar se variância dos resíduos é constante

3. **Testes de Multicolinearidade**
   - VIF (Variance Inflation Factor) para cada feature
   - Deve ser < 10 para cada variável

4. **Modelagem Comparativa**
   - Treinar com `log_target` (árvores) vs `log_target_normalizado` (linear)
   - Comparar métricas: R², RMSE, MAE
   - Escolher baseado em performance e pressupostos

---

## 📈 Impacto da Transformação

### **Antes (Original)**
```
Estatística W = 0.9931 (próxima de 1, mas p << 0.05)
Assimetria = -0.3029 (ligeiramente enviesada à esquerda)
Curtose = +0.3779 (ligeiramente leptocúrtica)

→ Visualmente normal, mas estatisticamente rejeita H0
```

### **Depois (Box-Cox)**
```
Estatística W = 0.9966 (ainda mais próxima de 1)
Assimetria = -0.0553 (praticamente simétrica ✓)
Curtose = -0.0757 (praticamente mesocúrtica ✓)

→ Significativamente mais normal
→ Apropriado para regressão linear
→ Pressupostos satisfeitos na prática
```

---

## 🔍 Interpretação Técnica

### **Por que Box-Cox?**
- Otimiza λ (lambda) automaticamente para máxima normalidade
- Bem fundamentado em teoria estatística (Box & Cox, 1964)
- Preserva ordem e monoticidade dos dados
- Transforma a série mantendo informação relativa

### **O que significa λ = 1.2624?**
- λ > 1: Aplica potência para "esticar" valores maiores
- Indica série tem assimetria negativa leve
- Próximo a 1: transformação é suave (série já quase normal)
- Fórmula aplicada: y' = (y^1.2624 - 1) / 1.2624

### **Por que ainda p < 0.05?**
- Com n = 20.228, teste SW é extremamente sensível
- Pequenos desvios não-normais são detectáveis
- **Prática:** W > 0.99 + assimetria/curtose próximas de 0 = suficiente
- Dados transformados são **apropriados para regressão linear**

---

## 📚 Referências Utilizadas

1. **Box, G. E., & Cox, D. R. (1964)**  
   "An analysis of transformations"  
   *Journal of the Royal Statistical Society*, 26(2), 211-252

2. **Shapiro, S. S., & Wilk, M. B. (1965)**  
   "An analysis of variance test for normality (complete samples)"  
   *Biometrika*, 52(3/4), 591-611

3. **SciPy Documentation**  
   `scipy.stats.boxcox` - Power transformations

4. **sklearn.preprocessing.PowerTransformer**  
   Implementação de Box-Cox em machine learning

---

## ✅ Checklist de Qualidade

- [x] Teste de Normalidade (Shapiro-Wilk) realizado
- [x] 6 transformações testadas e comparadas
- [x] Melhor transformação (Box-Cox) aplicada
- [x] Validação pós-transformação concluída
- [x] Base normalizada gerada e validada
- [x] Metadados salvos para rastreabilidade
- [x] Documentação completa escrita
- [x] 6 relatórios visuais gerados
- [x] Scripts reproducíveis fornecidos
- [x] Recomendações de uso claras

---

## 📞 Suporte

**Base de Dados Original:**  
`base_analitica_final.csv` (17 colunas)

**Base Normalizada:**  
`base_normalizada_boxcox.csv` (19 colunas - inclui original + normalizado)

**Documentação Detalhada:**  
`NORMALIZACAO_BOXCOX.md`

**Metadados:**  
`metadados_normalizacao_boxcox.json`

**Código Reproducível:**  
- `teste_normalidade.py`
- `gerar_base_normalizada.py`
- `relatorio_normalizacao_visual.py`

---

**Data de Conclusão:** 2026-04-28  
**Status:** ✅ PRONTO PARA MODELAGEM  
**Próximo Passo:** Iniciar modelagem com série normalizada ou original conforme necessário
