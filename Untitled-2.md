# Transparência Pública de Criciúma/SC
## Fatores que determinam e permitem prever pagamentos de empenhos com pessoal

**Subtítulo:** Análise de dados públicos municipais (2012–2026)

**Objetivo:** Identificar os fatores que influenciam o valor efetivamente pago dos empenhos de pessoal e avaliar sua capacidade preditiva para apoiar a gestão fiscal municipal.

---

# Slide 2 — Problema de Negócio

## Pergunta Central

> Quais fatores explicam o valor efetivamente pago dos empenhos de pessoal e como prever esse valor?

## Relevância para a Prefeitura

- Planejamento do fluxo de caixa
- Controle da despesa com pessoal
- Cumprimento da Lei de Responsabilidade Fiscal (LRF)
- Apoio à elaboração orçamentária

## Resultado Esperado

Construir uma base analítica capaz de antecipar pagamentos futuros a partir das características dos empenhos.

---

# Slide 3 — Base de Dados Utilizada

## Fontes Integradas

| Fonte | Registros |
|---------|---------:|
| Despesas com Pessoal | 20.691 |
| Programas e Ações | 3.750 |
| Cargos e Vencimentos | 51.582 |
| Adiantamentos | 1.053 |

## Base Final

- 20.228 registros válidos
- Série histórica de 2012 a 2026
- Dados públicos do Portal da Transparência

## Destaque

Base suficiente para análises estatísticas robustas e modelagem preditiva.

---

# Slide 4 — Construção da Base Analítica

## Processo de Integração

```text
Despesas com Pessoal
         ↓
 Programas e Ações
         ↓
 Cargos e Vencimentos
         ↓
   Adiantamentos
         ↓
 Base Analítica Final
```

## Cuidados Aplicados

- Validação das chaves de junção
- Ausência de duplicidades
- Sem explosão de registros
- Tratamento de valores nulos

## Resultado

Base consolidada e consistente para análise.

---

# Slide 5 — Preparação dos Dados

## Variável-Alvo

Valor efetivamente pago do empenho:

\[
log(1 + valorPagoEmpenho)
\]

## Motivos para Utilizar Logaritmo

- Redução da assimetria
- Tratamento indireto de outliers
- Melhor comportamento estatístico

## Engenharia de Atributos

Foram geradas:

**135 variáveis candidatas**

Categorias:

- Temporais
- Orçamentárias
- Recursos Humanos
- Contagens
- Lags temporais
- Target Encoding

---

# Slide 6 — Seleção das Variáveis

## Critério Utilizado

Correlação com o valor pago transformado em log.

## Resultado

```text
135 variáveis candidatas
          ↓
15 variáveis selecionadas
          ↓
Maior capacidade explicativa
```

## Principais Categorias Selecionadas

- Elemento de despesa
- Programa
- Órgão
- Unidade gestora

## Insight

As características estruturais do gasto foram mais importantes do que fatores temporais.

---

# Slide 7 — Principais Variáveis Explicativas

## Ranking das Variáveis

| Variável | Correlação |
|-----------|----------:|
| Unidade × Elemento (média) | 0,446 |
| Órgão × Programa × Elemento | 0,428 |
| Programa × Elemento | 0,424 |
| Unidade × Elemento (mediana) | 0,418 |
| Órgão × Programa × Elemento (mediana) | 0,408 |

## Interpretação

O contexto administrativo onde a despesa ocorre possui forte relação com o valor pago.

---

# Slide 8 — Principal Descoberta

## O Elemento de Despesa é o Principal Driver

Exemplos:

- Vencimentos
- Obrigações Patronais
- Diárias
- Material de Consumo

## Evidência

A combinação entre:

- Elemento
- Órgão
- Programa
- Unidade

explica grande parte da variação dos pagamentos.

## Mensagem Executiva

> O valor pago é determinado principalmente pela natureza da despesa e pelo contexto organizacional onde ela ocorre.

---

# Slide 9 — Variáveis com Baixa Relevância

## Fatores que Não Explicaram Bem os Pagamentos

### Temporais

- Mês
- Trimestre
- Dia do mês

### Operacionais

- Número de liquidações
- Número de documentos fiscais

### Administrativos

- Indicadores de RH
- Adiantamentos
- Valores agregados de programas

## Conclusão

Pagamentos de pessoal apresentam comportamento relativamente estável ao longo do tempo.

---

# Slide 10 — Limitações e Cuidados

## Multicolinearidade

As variáveis selecionadas apresentam forte correlação entre si.

Exemplos:

- Programa × Elemento
- Órgão × Elemento
- Unidade × Elemento

## Impacto

### Modelos Lineares

- Maior instabilidade
- Redundância de informação

### Modelos de Árvore

- Menor sensibilidade
- Melhor aproveitamento das interações

## Recomendação

Utilizar:

- XGBoost
- LightGBM

---

# Slide 11 — Conclusões e Próximos Passos

## Principais Conclusões

✅ Base consolidada com mais de 20 mil registros

✅ Elemento de despesa é o principal determinante do pagamento

✅ Contexto organizacional aumenta significativamente o poder preditivo

✅ Variáveis temporais apresentaram pouca influência

## Próximas Evoluções

- Target Encoding Out-of-Fold
- Smoothing Bayesiano
- LightGBM/XGBoost
- NLP em histórico dos empenhos
- Validação temporal prospectiva

## Mensagem Final

> Os pagamentos de pessoal do município apresentam padrões estruturais consistentes e previsíveis, permitindo o desenvolvimento de modelos capazes de apoiar o planejamento fiscal e orçamentário da Prefeitura de Criciúma.

---

# Slide Final — Perguntas

## Obrigado!

**Contato**
- Nome
- Curso/Disciplina
- Instituição

### Perguntas?