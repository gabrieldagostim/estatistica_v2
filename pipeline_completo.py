"""
==========================================================================
PIPELINE COMPLETO — Trabalho de Extensao Estatistica Aplicada
Transparencia Municipio de Criciuma/SC (2012-2026)
==========================================================================

Arquivo unico que reproduz toda a entrega:
  1. Carga dos JSONs                    -> dados/Despesas com Pessoal-*.json
  2. Feature engineering p/ OLS         -> log_target, log_valorEmpenhado, etc.
  3. Teste de normalidade (Shapiro-Wilk) + Box-Cox
  4. Calculo do tamanho amostral
  5. Ajuste OLS (statsmodels) + pressupostos (VIF, Breusch-Pagan, SW residuos)
  6. ESTIMATIVAS PONTUAIS e INTERVALARES  (CHECKPOINT 3)
  7. PREVISOES OLS PARA INTERVALO FUTURO (CHECKPOINT 3)
  8. Persistencia de bases, modelo, JSONs e graficos

Execucao:
    python pipeline_completo.py

Saidas:
    bases/base_ols.csv
    resultados/checkpoint1_resultado.json
    resultados/checkpoint2_ols_summary.txt
    resultados/checkpoint2_metricas.json
    resultados/checkpoint2_modelo.pkl
    resultados/checkpoint3_estimativas.json
    resultados/relatorio_final.json
    graficos/pipeline_01_normalidade.png
    graficos/pipeline_02_diagnostico_ols.png
    graficos/pipeline_03_estimativas_previsoes.png
==========================================================================
"""
import os, json, glob, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from scipy.stats import shapiro, boxcox, probplot
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# CONFIGURACOES GERAIS
# --------------------------------------------------------------------------
DADOS_DIR       = 'dados'
BASES_DIR       = 'bases'
RESULTADOS_DIR  = 'resultados'
GRAFICOS_DIR    = 'graficos'
RANDOM_STATE    = 42

for d in (BASES_DIR, RESULTADOS_DIR, GRAFICOS_DIR):
    os.makedirs(d, exist_ok=True)


def header(titulo):
    print('\n' + '=' * 74)
    print(f'  {titulo}')
    print('=' * 74)


class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON que aceita tipos numpy/bool."""
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return super().default(obj)


# ==========================================================================
# BLOCO 1 — CARGA E FEATURE ENGINEERING PARA OLS
# ==========================================================================
header('BLOCO 1 — CARGA DOS DADOS + FEATURE ENGINEERING (OLS)')

def carrega_json(prefixo):
    frames = []
    for f in sorted(glob.glob(os.path.join(DADOS_DIR, f'{prefixo}*.json'))):
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        frames.append(pd.json_normalize(data, max_level=1))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


print(f'Carregando JSONs de: {DADOS_DIR}/Despesas com Pessoal-*.json')
dp = carrega_json('Despesas com Pessoal')
print(f'  Bruto: {len(dp):,} registros / {dp.shape[1]} colunas')

# Tipos
dp['dataEmpenho']  = pd.to_datetime(dp['dataEmpenho'], errors='coerce')
dp['anoExercicio'] = pd.to_numeric(dp['anoExercicio'], errors='coerce')

num_cols = ['valorEmpenho', 'valorEmpenhado', 'valorPagoEmpenho',
            'valorLiquidadoEmpenho', 'saldoAPagar', 'saldoALiquidar',
            'ValorAnuladoEmpenho']
for c in num_cols:
    if c in dp.columns:
        dp[c] = pd.to_numeric(dp[c], errors='coerce').fillna(0)

# Filtro de alvo valido
dp = dp[dp['valorPagoEmpenho'] > 0].copy()
print(f'  Apos filtro (valorPagoEmpenho > 0): {len(dp):,} registros')

# Variavel-alvo
TARGET = 'valorPagoEmpenho'
dp['log_target'] = np.log1p(dp[TARGET])

# Features numericas
dp['log_valorEmpenhado'] = np.log1p(dp['valorEmpenhado'].clip(lower=0))
dp['log_saldoAPagar']    = np.log1p(dp['saldoAPagar'].clip(lower=0))
dp['n_pagamentos']       = dp['pagamentos'].apply(lambda x: len(x) if isinstance(x, list) else 0)
dp['ano_empenho']        = dp['dataEmpenho'].dt.year
dp['mes_empenho']        = dp['dataEmpenho'].dt.month
dp['trimestre']          = dp['dataEmpenho'].dt.quarter

# Dummies trimestrais (referencia = Q1)
for q in [2, 3, 4]:
    dp[f'trim_{q}'] = (dp['trimestre'] == q).astype(int)

# Categorica: elemento de despesa (top 7 + Outros)
if 'descricaoElemento' in dp.columns:
    top_elem = dp['descricaoElemento'].value_counts().nlargest(7).index.tolist()
    dp['elemento_cat'] = dp['descricaoElemento'].apply(
        lambda x: x if x in top_elem else 'Outros'
    )
else:
    dp['elemento_cat'] = 'Unico'

# Categorica: tipo de empenho
if 'tipoEmpenho' in dp.columns:
    dp['tipo_empenho'] = dp['tipoEmpenho'].fillna('Desconhecido').astype(str)
else:
    dp['tipo_empenho'] = 'Unico'

# Selecionar colunas finais
colunas_ols = [
    'valorPagoEmpenho', 'log_target',
    'log_valorEmpenhado', 'log_saldoAPagar', 'n_pagamentos',
    'ano_empenho', 'mes_empenho', 'trimestre',
    'trim_2', 'trim_3', 'trim_4',
    'elemento_cat', 'tipo_empenho',
]
colunas_ols = [c for c in colunas_ols if c in dp.columns]
df = dp[colunas_ols].dropna(subset=['log_target', 'log_valorEmpenhado']).copy()
df.to_csv(os.path.join(BASES_DIR, 'base_ols.csv'), index=False)
print(f'\nBase OLS salva: {BASES_DIR}/base_ols.csv  ({df.shape[0]:,} regs x {df.shape[1]} cols)')


# ==========================================================================
# BLOCO 2 — NORMALIDADE (SHAPIRO-WILK) + BOX-COX + TAMANHO AMOSTRAL
# ==========================================================================
header('BLOCO 2 — NORMALIDADE, BOX-COX E TAMANHO AMOSTRAL')

N = len(df)
y_log  = df['log_target'].dropna().values
y_orig = df['valorPagoEmpenho'].dropna().values

# Estatisticas descritivas do alvo (log)
media_log   = float(np.mean(y_log))
mediana_log = float(np.median(y_log))
dp_log      = float(np.std(y_log, ddof=1))
skew_log    = float(stats.skew(y_log))
kurt_log    = float(stats.kurtosis(y_log))

print(f'log_target: N={N:,} | media={media_log:.4f} | dp={dp_log:.4f}')
print(f'           skew={skew_log:+.4f} | kurt={kurt_log:+.4f}')

# Shapiro-Wilk (amostra de 5000 — limite recomendado)
SW_N = min(5000, N)
np.random.seed(RANDOM_STATE)
idx_sw  = np.random.choice(N, SW_N, replace=False)
sw_orig = y_log[idx_sw]
W_orig, p_orig = shapiro(sw_orig)
print(f'\nShapiro-Wilk (original, n={SW_N}):  W={W_orig:.6f}  p={p_orig:.2e}')
print(f'  -> {"NAO normal (p<0.05); Box-Cox indicado" if p_orig < 0.05 else "Normal"}')

# Box-Cox (requer y > 0; log_target ja > 0 por construcao log1p e filtro > 0)
y_bc, lambda_bc = boxcox(y_log)
sw_bc = y_bc[idx_sw]
W_bc, p_bc = shapiro(sw_bc)
print(f'\nBox-Cox: lambda otimo = {lambda_bc:.4f}')
print(f'Shapiro-Wilk (Box-Cox, n={SW_N}): W={W_bc:.6f}  p={p_bc:.2e}')

melhoria_p   = (p_bc / p_orig) if p_orig > 0 else np.nan
reducao_skew = (1 - abs(stats.skew(y_bc)) / (abs(skew_log) + 1e-9)) * 100
reducao_kurt = (1 - abs(stats.kurtosis(y_bc)) / (abs(kurt_log) + 1e-9)) * 100
print(f'  Melhoria no p-valor : {melhoria_p:,.2f}x')
print(f'  Reducao assimetria  : {reducao_skew:.1f}%')
print(f'  Reducao curtose     : {reducao_kurt:.1f}%')

# Adiciona Box-Cox a base (rastreabilidade)
df['log_target_bc'] = y_bc
df['lambda_boxcox'] = lambda_bc
df.to_csv(os.path.join(BASES_DIR, 'base_ols.csv'), index=False)

# Tamanho amostral
print('\n--- TAMANHO AMOSTRAL (n = z^2 * p(1-p) / E^2) ---')

def calcula_n(z, p, E):
    return math.ceil((z ** 2) * p * (1 - p) / (E ** 2))

def correcao_finita(n, N_pop):
    if n / N_pop > 0.05:
        return math.ceil(n / (1 + (n - 1) / N_pop))
    return n

z_95, z_99 = 1.96, 2.5758
p_max      = 0.5
cenarios_n = [
    ('IC 95%, E=5%', z_95, p_max, 0.05),
    ('IC 99%, E=5%', z_99, p_max, 0.05),
    ('IC 95%, E=1%', z_95, p_max, 0.01),
]
tamanho_amostral = []
for nome, z, p, E in cenarios_n:
    n_sem = calcula_n(z, p, E)
    n_com = correcao_finita(n_sem, N)
    sufic = N >= n_com
    print(f'  {nome:18s} -> n_sem={n_sem:>6} | n_corrigido={n_com:>6} | disp={N:,} | {"OK" if sufic else "INSUFIC."}')
    tamanho_amostral.append({'cenario': nome, 'z': z, 'p': p, 'E': E,
                             'n_sem_correcao': n_sem, 'n_corrigido': n_com,
                             'disponivel': N, 'suficiente': bool(sufic)})

n_necessario = correcao_finita(calcula_n(z_95, p_max, 0.05), N)
print(f'\nCenario principal: n_necessario={n_necessario} | disponivel={N:,} (suficiente)')

# Salva Checkpoint 1 (normalidade)
cp1 = {
    'n_registros': int(N),
    'variavel_alvo': 'log_target = log1p(valorPagoEmpenho)',
    'estatisticas': {
        'media': round(media_log, 4),
        'mediana': round(mediana_log, 4),
        'desvio_padrao': round(dp_log, 4),
        'assimetria': round(skew_log, 4),
        'curtose': round(kurt_log, 4),
    },
    'shapiro_wilk_original': {'W': round(W_orig, 6), 'p': float(p_orig),
                              'normal': bool(p_orig >= 0.05)},
    'boxcox': {
        'lambda_otimo': round(float(lambda_bc), 4),
        'shapiro_wilk_pos_bc': {'W': round(W_bc, 6), 'p': float(p_bc)},
        'melhoria_p_valor': float(melhoria_p),
        'reducao_assimetria_pct': float(reducao_skew),
        'reducao_curtose_pct': float(reducao_kurt),
    },
    'tamanho_amostral': tamanho_amostral,
    'n_necessario_principal': int(n_necessario),
}
with open(os.path.join(RESULTADOS_DIR, 'checkpoint1_resultado.json'), 'w', encoding='utf-8') as f:
    json.dump(cp1, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
print(f'Salvo: {RESULTADOS_DIR}/checkpoint1_resultado.json')


# ==========================================================================
# BLOCO 3 — AJUSTE OLS + VERIFICACAO DE PRESSUPOSTOS
# ==========================================================================
header('BLOCO 3 — REGRESSAO OLS (statsmodels) + PRESSUPOSTOS')

# Garantir tipos para a formula
df['elemento_cat'] = df['elemento_cat'].fillna('Outros').astype(str)
df['tipo_empenho'] = df['tipo_empenho'].fillna('Desconhecido').astype(str)

tem_elemento = df['elemento_cat'].nunique() > 1
tem_tipo     = df['tipo_empenho'].nunique() > 1

formula_partes = ['log_valorEmpenhado', 'n_pagamentos', 'ano_empenho',
                  'trim_2', 'trim_3', 'trim_4']
if tem_elemento:
    formula_partes.append('C(elemento_cat)')
if tem_tipo:
    formula_partes.append('C(tipo_empenho)')

FORMULA = 'log_target ~ ' + ' + '.join(formula_partes)
print(f'Formula: {FORMULA}')

df_treino, df_teste = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
print(f'Split: treino={len(df_treino):,} | teste={len(df_teste):,}')

modelo = smf.ols(FORMULA, data=df_treino).fit()

# Metricas
y_pred_log = modelo.predict(df_teste)
y_real_log = df_teste['log_target']
r2_treino  = float(modelo.rsquared)
r2_teste   = float(1 - np.sum((y_real_log - y_pred_log) ** 2) /
                       np.sum((y_real_log - y_real_log.mean()) ** 2))
rmse_log   = float(np.sqrt(np.mean((y_real_log - y_pred_log) ** 2)))
mae_log    = float(np.mean(np.abs(y_real_log - y_pred_log)))

print(f'\n--- METRICAS ---')
print(f'  R^2 treino : {r2_treino:.4f}')
print(f'  R^2 teste  : {r2_teste:.4f}')
print(f'  RMSE (log) : {rmse_log:.4f}')
print(f'  MAE  (log) : {mae_log:.4f}')
print(f'  F-stat     : {modelo.fvalue:.2f}  (p={modelo.f_pvalue:.2e})')

# Pressupostos
residuos = modelo.resid
fitted   = modelo.fittedvalues
res_sample = residuos.sample(min(5000, len(residuos)), random_state=RANDOM_STATE).values
W_res, p_res = shapiro(res_sample)
print(f'\n--- PRESSUPOSTOS ---')
print(f'  SW residuos (n={len(res_sample)}): W={W_res:.4f}  p={p_res:.2e}')
print(f'    {"OK" if p_res >= 0.05 else "Atencao: n grande detecta desvios pequenos"}')

# Breusch-Pagan
try:
    bp_lm, bp_p, _, _ = het_breuschpagan(residuos, modelo.model.exog)
    print(f'  Breusch-Pagan: LM={bp_lm:.4f}  p={bp_p:.4f}'
          f'  {"(homocedastico)" if bp_p >= 0.05 else "(heterocedastico)"}')
    bp_result = {'lm': float(bp_lm), 'p': float(bp_p),
                 'homoscedastico': bool(bp_p >= 0.05)}
except Exception as e:
    print(f'  Breusch-Pagan: erro ({e})')
    bp_result = {}

# VIF
vif_records = []
try:
    X_vif = modelo.model.exog
    nomes = modelo.model.exog_names
    for i, nome in enumerate(nomes):
        if nome == 'Intercept':
            continue
        v = variance_inflation_factor(X_vif, i)
        vif_records.append({'feature': nome, 'VIF': round(float(v), 3)})
    vif_records.sort(key=lambda r: -r['VIF'])
    vif_max = max(r['VIF'] for r in vif_records) if vif_records else float('nan')
    print(f'  VIF max = {vif_max:.2f}  ({"OK <10" if vif_max < 10 else "atencao >10"})')
except Exception as e:
    print(f'  VIF: erro ({e})')
    vif_max = float('nan')

# Salva summary e modelo
with open(os.path.join(RESULTADOS_DIR, 'checkpoint2_ols_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(modelo.summary().as_text())
modelo.save(os.path.join(RESULTADOS_DIR, 'checkpoint2_modelo.pkl'))
print(f'Salvo: {RESULTADOS_DIR}/checkpoint2_ols_summary.txt')
print(f'Salvo: {RESULTADOS_DIR}/checkpoint2_modelo.pkl')

cp2 = {
    'formula': FORMULA,
    'n_treino': int(len(df_treino)),
    'n_teste': int(len(df_teste)),
    'metricas': {'r2_treino': round(r2_treino, 4),
                 'r2_teste':  round(r2_teste, 4),
                 'rmse_log':  round(rmse_log, 4),
                 'mae_log':   round(mae_log, 4)},
    'f_statistic': {'value': round(float(modelo.fvalue), 2),
                    'p': float(modelo.f_pvalue)},
    'pressupostos': {
        'normalidade_residuos': {'W': round(float(W_res), 4),
                                 'p': float(p_res),
                                 'ok': bool(p_res >= 0.05)},
        'homocedasticidade': bp_result,
        'vif': vif_records,
        'vif_max': None if math.isnan(vif_max) else round(float(vif_max), 3),
    },
}
with open(os.path.join(RESULTADOS_DIR, 'checkpoint2_metricas.json'), 'w', encoding='utf-8') as f:
    json.dump(cp2, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
print(f'Salvo: {RESULTADOS_DIR}/checkpoint2_metricas.json')


# ==========================================================================
# BLOCO 4 — ESTIMATIVAS PONTUAIS E INTERVALARES (CHECKPOINT 3 — entregavel 1)
# ==========================================================================
header('BLOCO 4 — ESTIMATIVAS PONTUAIS E INTERVALARES')

media_orig   = float(np.mean(y_orig))
mediana_orig = float(np.median(y_orig))
dp_orig      = float(np.std(y_orig, ddof=1))
ep_media     = dp_orig / math.sqrt(len(y_orig))
ep_media_log = dp_log  / math.sqrt(len(y_log))

print('--- ESTIMATIVAS PONTUAIS ---')
print(f'  valorPagoEmpenho (R$):  media=R${media_orig:,.2f} | mediana=R${mediana_orig:,.2f}'
      f' | dp=R${dp_orig:,.2f} | EP_media=R${ep_media:,.2f}')
print(f'  log_target           :  media={media_log:.4f}    | mediana={mediana_log:.4f}'
      f'    | dp={dp_log:.4f}    | EP_media={ep_media_log:.6f}')

def ic_media(dados, confianca=0.95, populacao_finita=False, N_pop=None):
    n = len(dados)
    m = float(np.mean(dados))
    s = float(np.std(dados, ddof=1))
    ep = s / math.sqrt(n)
    alpha = 1 - confianca
    if n < 30:
        tc = stats.t.ppf(1 - alpha / 2, df=n - 1)
        metodo = 't-Student'
    else:
        tc = stats.norm.ppf(1 - alpha / 2)
        metodo = 'Normal (z)'
    fpc = 1.0
    if populacao_finita and N_pop:
        fpc = math.sqrt((N_pop - n) / (N_pop - 1))
        ep *= fpc
    margem = tc * ep
    return {'metodo': metodo, 'n': n, 'media': m, 'ep': ep,
            'tc': float(tc), 'fpc': float(fpc), 'margem': float(margem),
            'li': m - margem, 'ls': m + margem, 'confianca': confianca}

print('\n--- INTERVALOS DE CONFIANCA (95%) ---')

ic_log_95   = ic_media(y_log,  confianca=0.95)
ic_orig_95  = ic_media(y_orig, confianca=0.95)
ic_finita   = ic_media(y_log,  confianca=0.95, populacao_finita=True, N_pop=N)

print(f'  IC 95% media log_target          : [{ic_log_95["li"]:.4f}, {ic_log_95["ls"]:.4f}]'
      f' (margem={ic_log_95["margem"]:.6f}, {ic_log_95["metodo"]})')
print(f'  IC 95% media valorPagoEmpenho R$ : [R${ic_orig_95["li"]:,.2f}, R${ic_orig_95["ls"]:,.2f}]'
      f' (margem=R${ic_orig_95["margem"]:,.2f})')
print(f'  IC 95% media log com FPC(N={N:,}) : [{ic_finita["li"]:.4f}, {ic_finita["ls"]:.4f}]'
      f' (fpc={ic_finita["fpc"]:.6f})')

# IC para proporcao (empenhos acima de R$5.000)
LIMIAR = 5000
p_hat   = float((y_orig > LIMIAR).mean())
ep_prop = math.sqrt(p_hat * (1 - p_hat) / len(y_orig))
z95     = stats.norm.ppf(0.975)
li_p, ls_p = p_hat - z95 * ep_prop, p_hat + z95 * ep_prop
print(f'  IC 95% prop. empenhos > R${LIMIAR:,}  : {p_hat*100:.2f}% '
      f'[{li_p*100:.2f}%, {ls_p*100:.2f}%]')

# IC para diferenca de medias (top 2 elementos)
ic_diff = None
if df['elemento_cat'].nunique() >= 2:
    top2 = df['elemento_cat'].value_counts().head(2).index.tolist()
    g1 = df[df['elemento_cat'] == top2[0]]['log_target'].values
    g2 = df[df['elemento_cat'] == top2[1]]['log_target'].values
    t_stat, p_ttest = stats.ttest_ind(g1, g2, equal_var=False)
    diff = float(np.mean(g1) - np.mean(g2))
    ep_diff = math.sqrt(np.var(g1, ddof=1) / len(g1) + np.var(g2, ddof=1) / len(g2))
    margem_diff = z95 * ep_diff
    ic_diff = {'g1': top2[0], 'g2': top2[1],
               'diferenca': round(diff, 4),
               'li': round(diff - margem_diff, 4),
               'ls': round(diff + margem_diff, 4),
               'p_ttest': float(p_ttest),
               'significativo_5pct': bool(p_ttest < 0.05)}
    print(f'  IC 95% diferenca medias [{top2[0][:25]} vs {top2[1][:25]}]:'
          f' {diff:+.4f}  [{ic_diff["li"]:+.4f}, {ic_diff["ls"]:+.4f}]'
          f'  t-test p={p_ttest:.2e}'
          f'  ({"signif." if p_ttest < 0.05 else "nao signif."})')


# ==========================================================================
# BLOCO 5 — PREVISOES OLS PARA INTERVALO FUTURO (CHECKPOINT 3 — entregavel 2)
# ==========================================================================
header('BLOCO 5 — PREVISOES OLS PARA INTERVALO FUTURO (IC pred. 95%)')

ano_max = int(df['ano_empenho'].max())
ano_futuro = ano_max + 1
print(f'Ano base maximo na base: {ano_max} -> previsoes para {ano_futuro}')

# Valores de referencia (medianas)
log_emp_med = float(df['log_valorEmpenhado'].median())
n_pag_med   = float(df['n_pagamentos'].median())
top_elem    = df['elemento_cat'].value_counts().head(3).index.tolist()
tipo_pad    = df['tipo_empenho'].value_counts().index[0]

cenarios_prev = []
trimestres_def = [
    (1, 0, 0, 0),
    (2, 1, 0, 0),
    (3, 0, 1, 0),
    (4, 0, 0, 1),
]
for elem in top_elem:
    for trim, t2, t3, t4 in trimestres_def:
        cenarios_prev.append({
            'elemento_cat': elem,
            'tipo_empenho': tipo_pad,
            'log_valorEmpenhado': log_emp_med,
            'n_pagamentos': n_pag_med,
            'ano_empenho': ano_futuro,
            'mes_empenho': trim * 3,
            'trimestre': trim,
            'trim_2': t2, 'trim_3': t3, 'trim_4': t4,
        })
df_prev_in = pd.DataFrame(cenarios_prev)

predicoes = modelo.get_prediction(df_prev_in)
pred_frame = predicoes.summary_frame(alpha=0.05)  # IC predicao 95% (obs_ci_*)

resultados_prev = []
print(f'\n{"Elemento":<32} {"Trim":<5} {"Prev.(log)":>11} {"Prev.(R$)":>14}'
      f'  {"IC Pred. 95% (R$)":>30}')
print('-' * 100)
for i, row in df_prev_in.iterrows():
    pf = pred_frame.iloc[i]
    mean_log = float(pf['mean'])
    li_log   = float(pf['obs_ci_lower'])
    ls_log   = float(pf['obs_ci_upper'])
    real     = math.expm1(mean_log)
    li_real  = math.expm1(li_log)
    ls_real  = math.expm1(ls_log)
    print(f'{row["elemento_cat"][:30]:<32} Q{int(row["trimestre"])}    '
          f'{mean_log:>11.4f} R$ {real:>11,.0f}'
          f'  [R$ {li_real:>11,.0f}, R$ {ls_real:>11,.0f}]')
    resultados_prev.append({
        'ano': ano_futuro,
        'trimestre': int(row['trimestre']),
        'elemento': row['elemento_cat'],
        'tipo_empenho': row['tipo_empenho'],
        'previsao_log': round(mean_log, 4),
        'ic_pred_log_li': round(li_log, 4),
        'ic_pred_log_ls': round(ls_log, 4),
        'previsao_real_reais': round(real, 2),
        'ic_pred_real_li': round(li_real, 2),
        'ic_pred_real_ls': round(ls_real, 2),
    })

# Persistencia CP3
cp3 = {
    'estimativas_pontuais': {
        'valorPagoEmpenho': {
            'media':         round(media_orig, 2),
            'mediana':       round(mediana_orig, 2),
            'desvio_padrao': round(dp_orig, 2),
            'erro_padrao_media': round(ep_media, 2),
        },
        'log_target': {
            'media':         round(media_log, 4),
            'mediana':       round(mediana_log, 4),
            'desvio_padrao': round(dp_log, 4),
            'erro_padrao_media': round(ep_media_log, 6),
        },
    },
    'intervalos_confianca': {
        'ic_media_log_95':   {'li': round(ic_log_95['li'], 4),
                              'ls': round(ic_log_95['ls'], 4),
                              'margem': round(ic_log_95['margem'], 6)},
        'ic_media_real_95':  {'li': round(ic_orig_95['li'], 2),
                              'ls': round(ic_orig_95['ls'], 2),
                              'margem': round(ic_orig_95['margem'], 2)},
        'ic_media_finita_95': {'li': round(ic_finita['li'], 4),
                               'ls': round(ic_finita['ls'], 4),
                               'fpc': round(ic_finita['fpc'], 6)},
        'ic_proporcao_95': {'limiar': LIMIAR,
                            'p_hat': round(p_hat, 4),
                            'li_pct': round(li_p * 100, 2),
                            'ls_pct': round(ls_p * 100, 2)},
        'ic_diferenca_medias': ic_diff,
    },
    'previsoes_ols_futuro': resultados_prev,
    'ano_previsto': ano_futuro,
}
with open(os.path.join(RESULTADOS_DIR, 'checkpoint3_estimativas.json'), 'w', encoding='utf-8') as f:
    json.dump(cp3, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
print(f'\nSalvo: {RESULTADOS_DIR}/checkpoint3_estimativas.json')


# ==========================================================================
# BLOCO 6 — GRAFICOS CONSOLIDADOS
# ==========================================================================
header('BLOCO 6 — GERANDO GRAFICOS')

# --- Grafico 1: Normalidade + Box-Cox -----------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Pipeline — Normalidade do alvo + Box-Cox', fontsize=13, fontweight='bold')

ax = axes[0, 0]
ax.hist(sw_orig, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
xmin, xmax = ax.get_xlim()
xr = np.linspace(xmin, xmax, 200)
ax.plot(xr, stats.norm.pdf(xr, np.mean(sw_orig), np.std(sw_orig)), 'r-', lw=2, label='Normal teorica')
ax.set_title('Histograma log_target (original)')
ax.legend(fontsize=8)

ax = axes[0, 1]
probplot(sw_orig, dist='norm', plot=ax)
ax.set_title(f'QQ-Plot Original  W={W_orig:.4f} p={p_orig:.2e}')

ax = axes[0, 2]
ax.boxplot(sw_orig, vert=True, patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.7))
ax.set_title('Boxplot Original')

ax = axes[1, 0]
ax.hist(sw_bc, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
xmin, xmax = ax.get_xlim()
xr = np.linspace(xmin, xmax, 200)
ax.plot(xr, stats.norm.pdf(xr, np.mean(sw_bc), np.std(sw_bc)), 'r-', lw=2, label='Normal teorica')
ax.set_title(f'Histograma Box-Cox (lambda={lambda_bc:.4f})')
ax.legend(fontsize=8)

ax = axes[1, 1]
probplot(sw_bc, dist='norm', plot=ax)
ax.set_title(f'QQ-Plot Box-Cox  W={W_bc:.4f} p={p_bc:.2e}')

ax = axes[1, 2]
ax.axis('off')
tab = [['Metrica', 'Original', 'Box-Cox'],
       ['W (SW)',    f'{W_orig:.4f}', f'{W_bc:.4f}'],
       ['p-valor',   f'{p_orig:.2e}', f'{p_bc:.2e}'],
       ['Assimetria', f'{skew_log:+.4f}', f'{stats.skew(y_bc):+.4f}'],
       ['Curtose',    f'{kurt_log:+.4f}', f'{stats.kurtosis(y_bc):+.4f}'],
       ['Lambda',    '-', f'{lambda_bc:.4f}']]
t = ax.table(cellText=tab[1:], colLabels=tab[0], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.2, 1.5)
ax.set_title('Comparativo', fontsize=10, pad=15)

plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, 'pipeline_01_normalidade.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Salvo: {GRAFICOS_DIR}/pipeline_01_normalidade.png')

# --- Grafico 2: Diagnostico OLS -----------------------------------------
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Pipeline — Diagnostico do Modelo OLS', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(fitted, residuos, alpha=0.3, s=10, color='steelblue')
ax1.axhline(0, color='red', lw=1)
ax1.set_xlabel('Valores ajustados'); ax1.set_ylabel('Residuos')
ax1.set_title('Residuos vs Ajustados')

ax2 = fig.add_subplot(gs[0, 1])
probplot(res_sample, dist='norm', plot=ax2)
ax2.set_title(f'QQ Residuos  W={W_res:.4f} p={p_res:.2e}')

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(res_sample, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
xr = np.linspace(res_sample.min(), res_sample.max(), 200)
ax3.plot(xr, stats.norm.pdf(xr, np.mean(res_sample), np.std(res_sample)), 'r-', lw=2)
ax3.set_title('Distribuicao dos residuos')

ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_real_log, y_pred_log, alpha=0.2, s=8, color='steelblue')
lmin = min(y_real_log.min(), y_pred_log.min())
lmax = max(y_real_log.max(), y_pred_log.max())
ax4.plot([lmin, lmax], [lmin, lmax], 'r-', lw=1.5)
ax4.set_xlabel('log_target real'); ax4.set_ylabel('log_target previsto')
ax4.set_title(f'Real vs. Previsto  R2_treino={r2_treino:.3f}  R2_teste={r2_teste:.3f}')

ax5 = fig.add_subplot(gs[1, 1])
coefs = modelo.params.drop('Intercept', errors='ignore')
pvals = modelo.pvalues.drop('Intercept', errors='ignore')
sig   = coefs[pvals < 0.05]
ci    = modelo.conf_int().drop('Intercept', errors='ignore')
ci_sig = ci.loc[sig.index]
err   = (ci_sig[1] - ci_sig[0]) / 2
cores = ['green' if v > 0 else 'red' for v in sig.values]
ax5.barh(range(len(sig)), sig.values, xerr=err.values, color=cores, alpha=0.7, capsize=3)
ax5.set_yticks(range(len(sig)))
ax5.set_yticklabels([n[:22] for n in sig.index], fontsize=7)
ax5.axvline(0, color='black', lw=0.8)
ax5.set_title(f'Coef. signif. (p<0.05) — {len(sig)}/{len(coefs)}')

ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
linhas = [
    ['Metrica', 'Valor'],
    ['N treino',   f'{len(df_treino):,}'],
    ['N teste',    f'{len(df_teste):,}'],
    ['R^2 treino', f'{r2_treino:.4f}'],
    ['R^2 teste',  f'{r2_teste:.4f}'],
    ['RMSE log',   f'{rmse_log:.4f}'],
    ['MAE log',    f'{mae_log:.4f}'],
    ['F-stat p',   f'{modelo.f_pvalue:.2e}'],
    ['VIF max',    f'{vif_max:.2f}'],
    ['SW resid p', f'{p_res:.2e}'],
    ['BP p',       f'{bp_result.get("p", float("nan")):.4f}'],
]
t = ax6.table(cellText=linhas[1:], colLabels=linhas[0], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.2, 1.4)
ax6.set_title('Resumo do modelo OLS', fontsize=10, pad=15)

plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, 'pipeline_02_diagnostico_ols.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Salvo: {GRAFICOS_DIR}/pipeline_02_diagnostico_ols.png')

# --- Grafico 3: Estimativas + Previsoes ---------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Pipeline — Estimativas (CP3 - 1) e Previsoes OLS Futuras (CP3 - 2)',
             fontsize=13, fontweight='bold')

# Distribuicao + IC para a media
ax = axes[0, 0]
ax.hist(y_log, bins=60, density=True, alpha=0.6, color='steelblue', edgecolor='white')
ax.axvline(media_log, color='red', lw=2, label=f'Media={media_log:.3f}')
ax.axvline(ic_log_95['li'], color='orange', lw=1.5, ls='--',
           label=f'IC95%: [{ic_log_95["li"]:.3f}, {ic_log_95["ls"]:.3f}]')
ax.axvline(ic_log_95['ls'], color='orange', lw=1.5, ls='--')
ax.set_title('log_target — distribuicao + IC 95% para a media')
ax.set_xlabel('log_target'); ax.legend(fontsize=8)

# IC por elemento (top 6)
ax = axes[0, 1]
top6 = df['elemento_cat'].value_counts().head(6).index.tolist()
medias_e, margens_e = [], []
for e in top6:
    g = df[df['elemento_cat'] == e]['log_target'].values
    ic_e = ic_media(g, confianca=0.95)
    medias_e.append(ic_e['media']); margens_e.append(ic_e['margem'])
ax.barh(range(len(top6)), medias_e, xerr=margens_e, color='steelblue', alpha=0.7, capsize=4)
ax.set_yticks(range(len(top6)))
ax.set_yticklabels([e[:25] for e in top6], fontsize=8)
ax.set_title('IC 95% media (log_target) por elemento')
ax.set_xlabel('media (log)')

# Previsoes futuras (errorbar)
ax = axes[1, 0]
df_p = pd.DataFrame(resultados_prev)
labels = [f"{r['elemento'][:18]}\nQ{r['trimestre']}" for _, r in df_p.iterrows()]
yerr_l = df_p['previsao_real_reais'] - df_p['ic_pred_real_li']
yerr_h = df_p['ic_pred_real_ls'] - df_p['previsao_real_reais']
x_pos = range(len(df_p))
ax.errorbar(x_pos, df_p['previsao_real_reais'], yerr=[yerr_l, yerr_h],
            fmt='o', capsize=5, color='coral', markersize=6)
ax.set_xticks(list(x_pos))
ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
ax.set_title(f'Previsoes OLS para {ano_futuro} c/ IC predicao 95%')
ax.set_ylabel('valorPagoEmpenho previsto (R$)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'R${x:,.0f}'))

# Tabela resumo
ax = axes[1, 1]
ax.axis('off')
linhas = [
    ['Indicador', 'Valor'],
    ['Media valorPago (R$)',  f'R$ {media_orig:,.2f}'],
    ['Mediana (R$)',          f'R$ {mediana_orig:,.2f}'],
    ['Desvio padrao (R$)',    f'R$ {dp_orig:,.2f}'],
    ['EP media (R$)',         f'R$ {ep_media:,.2f}'],
    ['IC 95% media (log)',    f'[{ic_log_95["li"]:.3f}, {ic_log_95["ls"]:.3f}]'],
    ['IC 95% media (R$)',     f'[R${ic_orig_95["li"]:,.0f}, R${ic_orig_95["ls"]:,.0f}]'],
    [f'Prop. > R${LIMIAR:,}',
        f'{p_hat*100:.1f}% [{li_p*100:.1f}%, {ls_p*100:.1f}%]'],
    ['Ano previsto',          str(ano_futuro)],
    ['Cenarios previstos',    str(len(resultados_prev))],
]
t = ax.table(cellText=linhas[1:], colLabels=linhas[0], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1.2, 1.5)
ax.set_title('Resumo Checkpoint 3', fontsize=10, pad=15)

plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS_DIR, 'pipeline_03_estimativas_previsoes.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f'Salvo: {GRAFICOS_DIR}/pipeline_03_estimativas_previsoes.png')


# ==========================================================================
# RELATORIO FINAL CONSOLIDADO
# ==========================================================================
header('RELATORIO FINAL — CONSOLIDADO')

relatorio = {
    'meta': {
        'projeto': 'Trabalho de Extensao — Estatistica Aplicada — Criciuma/SC',
        'arquivos_entrada': sorted(glob.glob(os.path.join(DADOS_DIR, 'Despesas com Pessoal-*.json'))),
        'n_registros': int(N),
        'variavel_alvo_bruta': TARGET,
        'variavel_alvo_modelada': 'log_target = log1p(valorPagoEmpenho)',
    },
    'checkpoint1_normalidade': cp1,
    'checkpoint2_ols': cp2,
    'checkpoint3_estimativas_previsoes': cp3,
}
with open(os.path.join(RESULTADOS_DIR, 'relatorio_final.json'), 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
print(f'Salvo: {RESULTADOS_DIR}/relatorio_final.json')

print('\nRESUMO EXECUTIVO')
print('-' * 60)
print(f'  Registros analisados : {N:,}')
print(f'  Tamanho amostral nec.: {n_necessario} (disp. {N:,}) -> OK')
print(f'  Box-Cox lambda       : {lambda_bc:.4f}')
print(f'  Formula OLS          : {FORMULA}')
print(f'  R^2 treino / teste   : {r2_treino:.4f} / {r2_teste:.4f}')
print(f'  VIF max              : {vif_max:.2f}')
print(f'  Estimativa pontual   : media = R$ {media_orig:,.2f}')
print(f'  IC 95% media (R$)    : [R${ic_orig_95["li"]:,.2f}, R${ic_orig_95["ls"]:,.2f}]')
print(f'  Previsoes p/ {ano_futuro}    : {len(resultados_prev)} cenarios gerados')

print('\nARTEFATOS GERADOS')
print('-' * 60)
print(f'  bases/base_ols.csv')
print(f'  resultados/checkpoint1_resultado.json')
print(f'  resultados/checkpoint2_ols_summary.txt')
print(f'  resultados/checkpoint2_metricas.json')
print(f'  resultados/checkpoint2_modelo.pkl')
print(f'  resultados/checkpoint3_estimativas.json')
print(f'  resultados/relatorio_final.json')
print(f'  graficos/pipeline_01_normalidade.png')
print(f'  graficos/pipeline_02_diagnostico_ols.png')
print(f'  graficos/pipeline_03_estimativas_previsoes.png')

print('\nPIPELINE EXECUTADA COM SUCESSO.')
