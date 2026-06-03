"""
Feature Engineering para OLS — sem target encodings
Gera bases/base_ols.csv com variaveis apropriadas para regressao linear
Trabalho de Extensao — Estatistica Aplicada — Transparencia Criciuma/SC
"""
import json, glob, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

DADOS = 'dados'
os.makedirs('bases', exist_ok=True)

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

print('Carregando dados...')
dp = carrega('Despesas com Pessoal')

dp['dataEmpenho'] = pd.to_datetime(dp['dataEmpenho'], errors='coerce')
dp['anoExercicio'] = pd.to_numeric(dp['anoExercicio'], errors='coerce')

num_cols = ['valorEmpenho','valorEmpenhado','valorPagoEmpenho','valorLiquidadoEmpenho',
            'saldoAPagar','saldoALiquidar','ValorAnuladoEmpenho',
            'valorRestosAPagarProcessados','valorRestosAPagarNaoProcessados']
for c in num_cols:
    if c in dp.columns:
        dp[c] = pd.to_numeric(dp[c], errors='coerce').fillna(0)

dp = dp[dp['valorPagoEmpenho'] > 0].copy()
print(f'Registros validos: {len(dp)}')

# -----------------------------------------------------------------------
# 2. TARGET
# -----------------------------------------------------------------------
dp['log_target'] = np.log1p(dp['valorPagoEmpenho'])

# -----------------------------------------------------------------------
# 3. FEATURES NUMERICAS
# -----------------------------------------------------------------------
dp['log_valorEmpenhado'] = np.log1p(dp['valorEmpenhado'].clip(lower=0))
dp['log_saldoAPagar']    = np.log1p(dp['saldoAPagar'].clip(lower=0))
dp['n_pagamentos']       = dp['pagamentos'].apply(lambda x: len(x) if isinstance(x, list) else 0)
dp['ano_empenho']        = dp['dataEmpenho'].dt.year
dp['mes_empenho']        = dp['dataEmpenho'].dt.month
dp['trimestre']          = dp['dataEmpenho'].dt.quarter

# dummies de trimestre (referencia = Q1)
for q in [2, 3, 4]:
    dp[f'trim_{q}'] = (dp['trimestre'] == q).astype(int)

# -----------------------------------------------------------------------
# 4. ELEMENTO DE DESPESA — top categorias + "Outros"
# -----------------------------------------------------------------------
if 'descricaoElemento' in dp.columns:
    top_elem = dp['descricaoElemento'].value_counts().nlargest(7).index.tolist()
    dp['elemento_cat'] = dp['descricaoElemento'].apply(
        lambda x: x if x in top_elem else 'Outros'
    )
    print(f'\nCategorias de elemento_cat:')
    print(dp['elemento_cat'].value_counts().to_string())
else:
    dp['elemento_cat'] = 'Unico'

# -----------------------------------------------------------------------
# 5. TIPO DE EMPENHO
# -----------------------------------------------------------------------
if 'tipoEmpenho' in dp.columns:
    dp['tipo_empenho'] = dp['tipoEmpenho'].fillna('Desconhecido')
    print(f'\nTipos de empenho:')
    print(dp['tipo_empenho'].value_counts().to_string())
else:
    dp['tipo_empenho'] = 'Unico'

# -----------------------------------------------------------------------
# 6. SELECIONAR E SALVAR
# -----------------------------------------------------------------------
colunas_finais = [
    'valorPagoEmpenho', 'log_target',
    'log_valorEmpenhado', 'log_saldoAPagar', 'n_pagamentos',
    'ano_empenho', 'mes_empenho', 'trimestre',
    'trim_2', 'trim_3', 'trim_4',
    'elemento_cat', 'tipo_empenho',
]
colunas_finais = [c for c in colunas_finais if c in dp.columns]

base_ols = dp[colunas_finais].dropna(subset=['log_target', 'log_valorEmpenhado'])
base_ols.to_csv('bases/base_ols.csv', index=False)
print(f'\nBase OLS salva: bases/base_ols.csv')
print(f'  Shape: {base_ols.shape}')
print(f'  Colunas: {list(base_ols.columns)}')

# -----------------------------------------------------------------------
# 7. CORRELACOES DA BASE OLS
# -----------------------------------------------------------------------
print('\n=== CORRELACAO COM log_target (base OLS) ===')
num_features = ['log_valorEmpenhado', 'log_saldoAPagar', 'n_pagamentos',
                'ano_empenho', 'mes_empenho', 'trimestre', 'trim_2', 'trim_3', 'trim_4']
num_features = [c for c in num_features if c in base_ols.columns]
corr_ols = base_ols[num_features + ['log_target']].corr()['log_target'].drop('log_target')
corr_ols = corr_ols.reindex(corr_ols.abs().sort_values(ascending=False).index)
for v, c in corr_ols.items():
    print(f'  {v:30s}: {c:+.4f}')

print('\n[OK] base_ols.csv gerada. Proximos passos:')
print('  python scripts/03_checkpoint1_normalidade.py')
