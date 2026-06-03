"""
EDA — Definicao da pergunta-problema e identificacao de 25+ variaveis candidatas
Trabalho de Extensao — Estatistica Aplicada — Transparencia Criciuma/SC
"""
import json, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

DADOS = 'dados'
os.makedirs('graficos', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

# -----------------------------------------------------------------------
# 1. CARREGAR DADOS
# -----------------------------------------------------------------------
def carrega(prefixo):
    frames = []
    for f in sorted(glob.glob(os.path.join(DADOS, f'{prefixo}*.json'))):
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        frames.append(pd.json_normalize(data, max_level=1))
    return pd.concat(frames, ignore_index=True)

print('Carregando Despesas com Pessoal...')
dp = carrega('Despesas com Pessoal')
print(f'  Total bruto: {len(dp)} registros, {dp.shape[1]} colunas')

# -----------------------------------------------------------------------
# 2. LIMPEZA BASICA
# -----------------------------------------------------------------------
dp['dataEmpenho'] = pd.to_datetime(dp['dataEmpenho'], errors='coerce')
dp['anoExercicio'] = pd.to_numeric(dp['anoExercicio'], errors='coerce')

num_cols = ['valorEmpenho','valorEmpenhado','valorPagoEmpenho','valorLiquidadoEmpenho',
            'saldoAPagar','saldoALiquidar','saldoAPagarLiquidado',
            'valorRestosAPagarProcessados','valorRestosAPagarNaoProcessados',
            'ValorAnuladoEmpenho','valorPagoRestosEmpenho']
for c in num_cols:
    if c in dp.columns:
        dp[c] = pd.to_numeric(dp[c], errors='coerce').fillna(0)

dp = dp[dp['valorPagoEmpenho'] > 0].copy()
print(f'  Apos filtro valorPagoEmpenho > 0: {len(dp)} registros')

# -----------------------------------------------------------------------
# 3. TARGET
# -----------------------------------------------------------------------
dp['log_target'] = np.log1p(dp['valorPagoEmpenho'])

# -----------------------------------------------------------------------
# 4. GERAR 28 VARIAVEIS CANDIDATAS
# -----------------------------------------------------------------------

# --- Temporais ---
dp['ano_empenho']   = dp['dataEmpenho'].dt.year
dp['mes_empenho']   = dp['dataEmpenho'].dt.month
dp['trimestre']     = dp['dataEmpenho'].dt.quarter
dp['dia_semana']    = dp['dataEmpenho'].dt.dayofweek
dp['dia_mes']       = dp['dataEmpenho'].dt.day
dp['fim_ano']       = (dp['mes_empenho'] >= 11).astype(int)
dp['inicio_ano']    = (dp['mes_empenho'] <= 2).astype(int)
dp['dia_util']      = (dp['dia_semana'] < 5).astype(int)

# --- Contagens de listas aninhadas ---
dp['n_pagamentos']  = dp['pagamentos'].apply(lambda x: len(x) if isinstance(x, list) else 0)
dp['n_liquidacoes'] = dp['liquidacoes'].apply(lambda x: len(x) if isinstance(x, list) else 0)
dp['n_docfiscais']  = dp['documentosFiscais'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# --- Numéricas / razões ---
dp['log_valorEmpenhado']   = np.log1p(dp['valorEmpenhado'].clip(lower=0))
dp['log_saldoAPagar']      = np.log1p(dp['saldoAPagar'].clip(lower=0))
dp['log_saldoALiquidar']   = np.log1p(dp['saldoALiquidar'].clip(lower=0))
dp['log_valorLiquidado']   = np.log1p(dp['valorLiquidadoEmpenho'].clip(lower=0))
dp['tx_anulacao']          = dp['ValorAnuladoEmpenho'] / (dp['valorEmpenhado'].abs() + 1)
dp['tx_restos']            = (dp['valorRestosAPagarProcessados'] + dp['valorRestosAPagarNaoProcessados']) / (dp['valorEmpenhado'].abs() + 1)

# --- Categoricas encodadas numericamente (para correlacao) ---
le = LabelEncoder()
for col in ['descricaoOrgao', 'descricaoUnidade', 'descricaoElemento',
            'descricaoFuncao', 'descricaoSubfuncao', 'descricaoPrograma',
            'tipoEmpenho', 'categoriaEmpenho']:
    if col in dp.columns:
        dp[f'enc_{col}'] = le.fit_transform(dp[col].fillna('DESCONHECIDO'))

# --- Target encodings para análise de correlação (media do target por categoria) ---
# Nota: target encodings são usados aqui APENAS para análise de correlação (CP1).
# O modelo OLS (CP2) usa dummies para evitar data leakage.
def te_encode(df, col, target, prefix):
    if col not in df.columns:
        return df
    g = df.groupby(col)[target].agg(['mean', 'median'])
    g.columns = [f'{prefix}_mean', f'{prefix}_median']
    return df.merge(g.reset_index(), on=col, how='left')

for col, prefix in [
    ('descricaoElemento', 'te_elemento'),
    ('descricaoOrgao', 'te_orgao'),
    ('descricaoUnidade', 'te_unidade'),
    ('descricaoFuncao', 'te_funcao'),
    ('descricaoSubfuncao', 'te_subfuncao'),
    ('descricaoPrograma', 'te_programa'),
    ('tipoEmpenho', 'te_tipo'),
]:
    dp = te_encode(dp, col, 'log_target', prefix)

# Encodings de interação (pares) — capturam efeitos combinados
def te_interacao(df, cols, target, prefix):
    chave = '_x_'.join(cols)
    df[chave] = df[cols[0]].fillna('').astype(str)
    for c in cols[1:]:
        df[chave] = df[chave] + '||' + df[c].fillna('').astype(str)
    return te_encode(df, chave, target, prefix)

dp = te_interacao(dp, ['descricaoUnidade', 'descricaoElemento'], 'log_target', 'te_uniEle')
dp = te_interacao(dp, ['descricaoOrgao', 'descricaoElemento'],   'log_target', 'te_orgEle')
dp = te_interacao(dp, ['descricaoPrograma', 'descricaoElemento'],'log_target', 'te_progEle')
dp = te_interacao(dp, ['descricaoFuncao', 'descricaoElemento'],  'log_target', 'te_funEle')
dp = te_interacao(dp, ['descricaoSubfuncao', 'descricaoElemento'],'log_target','te_subEle')
dp = te_interacao(dp, ['descricaoOrgao', 'descricaoPrograma', 'descricaoElemento'], 'log_target', 'te_orgProgEle')

candidatas = [
    # Temporais
    'ano_empenho', 'mes_empenho', 'trimestre', 'dia_semana', 'dia_mes',
    'fim_ano', 'inicio_ano',
    # Contagens
    'n_pagamentos', 'n_liquidacoes', 'n_docfiscais',
    # Numéricas financeiras
    'log_valorEmpenhado', 'log_saldoAPagar', 'log_saldoALiquidar',
    'log_valorLiquidado', 'tx_anulacao', 'tx_restos',
    # Label encodings (raw categories)
    'enc_descricaoOrgao', 'enc_descricaoElemento', 'enc_tipoEmpenho',
    # Target encodings simples por categoria
    'te_elemento_mean', 'te_elemento_median',
    'te_orgao_mean', 'te_unidade_mean', 'te_funcao_mean',
    # Target encodings de interação (pares e triplas)
    'te_uniEle_mean', 'te_uniEle_median',
    'te_orgEle_mean', 'te_orgEle_median',
    'te_progEle_mean', 'te_progEle_median',
    'te_funEle_mean', 'te_funEle_median',
    'te_subEle_mean', 'te_subEle_median',
    'te_orgProgEle_mean', 'te_orgProgEle_median',
]
candidatas = [c for c in candidatas if c in dp.columns]
print(f'\nTotal de variaveis candidatas: {len(candidatas)}')

# -----------------------------------------------------------------------
# 5. CORRELACOES
# -----------------------------------------------------------------------
print('\n=== CORRELACAO COM log_target ===')
corr = dp[candidatas + ['log_target']].corr()['log_target'].drop('log_target')
corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)

resultados = []
for v, c in corr_sorted.items():
    flag = '***' if abs(c) > 0.3 else ('**' if abs(c) > 0.1 else '  ')
    print(f'  {flag} {v:35s}: {c:+.4f}')
    resultados.append({'variavel': v, 'correlacao': round(c, 4), 'selecionada': abs(c) > 0.3})

n_selecionadas = sum(1 for r in resultados if r['selecionada'])
print(f'\nVariaveis com |r| > 0.3: {n_selecionadas}')

df_corr = pd.DataFrame(resultados)
df_corr.to_csv('resultados/candidatas_variaveis.csv', index=False)
print('Salvo: resultados/candidatas_variaveis.csv')

# -----------------------------------------------------------------------
# 6. GRAFICOS EDA
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('EDA — Despesas com Pessoal Criciuma/SC (2012-2026)', fontsize=14, fontweight='bold')

axes[0, 0].hist(dp['log_target'], bins=50, edgecolor='white', color='steelblue')
axes[0, 0].set_title('Distribuicao: log_target')
axes[0, 0].set_xlabel('log1p(valorPagoEmpenho)')

dp['ano_empenho'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title('Empenhos por Ano')
axes[0, 1].tick_params(axis='x', rotation=45)

dp['trimestre'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 2], color='coral')
axes[0, 2].set_title('Empenhos por Trimestre')

if 'descricaoElemento' in dp.columns:
    top_elem = dp['descricaoElemento'].value_counts().head(8)
    top_elem.plot(kind='barh', ax=axes[1, 0], color='steelblue')
    axes[1, 0].set_title('Top 8 Elementos de Despesa')
    axes[1, 0].tick_params(axis='y', labelsize=7)

top5_corr = corr_sorted.head(8)
colors = ['green' if v > 0 else 'red' for v in top5_corr.values]
axes[1, 1].barh(top5_corr.index, top5_corr.values, color=colors)
axes[1, 1].set_title('Top 8 Correlacoes com log_target')
axes[1, 1].axvline(0, color='black', linewidth=0.5)

if 'descricaoElemento' in dp.columns:
    top_elem_list = dp['descricaoElemento'].value_counts().head(6).index.tolist()
    dp_top = dp[dp['descricaoElemento'].isin(top_elem_list)]
    dp_top.boxplot(column='log_target', by='descricaoElemento', ax=axes[1, 2],
                   vert=True, patch_artist=True)
    axes[1, 2].set_title('log_target por Elemento (top 6)')
    axes[1, 2].tick_params(axis='x', rotation=45, labelsize=7)
    plt.sca(axes[1, 2])
    plt.title('log_target por Elemento')

plt.tight_layout()
plt.savefig('graficos/eda_01_visao_geral.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: graficos/eda_01_visao_geral.png')

# -----------------------------------------------------------------------
# 7. ESTATISTICAS DESCRITIVAS DA VARIAVEL-ALVO
# -----------------------------------------------------------------------
print('\n=== ESTATISTICAS DESCRITIVAS ===')
stats_target = dp['valorPagoEmpenho'].describe()
print(stats_target.to_string())
print(f'\nMédia (original): R$ {dp["valorPagoEmpenho"].mean():,.2f}')
print(f'Mediana (original): R$ {dp["valorPagoEmpenho"].median():,.2f}')
print(f'Desvio Padrao: R$ {dp["valorPagoEmpenho"].std():,.2f}')

print('\n=== VALORES UNICOS POR CATEGORICA ===')
for c in ['descricaoOrgao','descricaoElemento','descricaoFuncao','tipoEmpenho']:
    if c in dp.columns:
        print(f'  {c:30s}: {dp[c].nunique()} valores unicos')

print('\n[OK] EDA concluido. Proximos passos:')
print('  python scripts/02_features_ols.py')
