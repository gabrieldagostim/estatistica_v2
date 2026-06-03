"""
CHECKPOINT 2 — Tamanho Amostral + Regressao OLS (statsmodels)
Trabalho de Extensao — Estatistica Aplicada — Transparencia Criciuma/SC
"""
import os, json, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

os.makedirs('graficos', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

# -----------------------------------------------------------------------
# 1. CARREGAR BASE OLS
# -----------------------------------------------------------------------
BASE = 'bases/base_ols.csv'
if not os.path.exists(BASE):
    print(f'ERRO: {BASE} nao encontrado. Execute primeiro:')
    print('  python scripts/02_features_ols.py')
    exit(1)

df = pd.read_csv(BASE)
N = len(df)
print(f'Base carregada: {N} registros')

# Garantir que elemento_cat e tipo_empenho sao strings
for col in ['elemento_cat', 'tipo_empenho']:
    if col in df.columns:
        df[col] = df[col].fillna('Outros').astype(str)

# -----------------------------------------------------------------------
# 2. TAMANHO AMOSTRAL
# -----------------------------------------------------------------------
print('\n=== CALCULO DO TAMANHO AMOSTRAL ===')

def calcula_n(z, p, E):
    return math.ceil((z**2 * p * (1 - p)) / (E**2))

def correcao_finita(n, N):
    if n / N > 0.05:
        return math.ceil(n / (1 + (n - 1) / N))
    return n

z_95  = 1.9600
z_99  = 2.5758
p_max = 0.5
E_5   = 0.05
E_1   = 0.01

cenarios = [
    ('95% confianca, E=5%', z_95, p_max, E_5),
    ('99% confianca, E=5%', z_99, p_max, E_5),
    ('95% confianca, E=1%', z_95, p_max, E_1),
]

resultados_amostra = []
for nome, z, p, E in cenarios:
    n_sem = calcula_n(z, p, E)
    n_com = correcao_finita(n_sem, N)
    suficiente = N >= n_com
    print(f'  {nome}:')
    print(f'    n (sem correcao) = {n_sem}')
    print(f'    n (com correcao finita, N={N}) = {n_com}')
    print(f'    Disponivel: {N} >> {"Suficiente ✓" if suficiente else "Insuficiente ✗"}')
    resultados_amostra.append({'cenario': nome, 'z': z, 'p': p, 'E': E,
                                'n_sem_correcao': n_sem, 'n_corrigido': n_com,
                                'disponivel': N, 'suficiente': bool(suficiente)})

# Cenario padrao para o trabalho
n_necessario = correcao_finita(calcula_n(z_95, p_max, E_5), N)
print(f'\nCenario principal (95%, E=5%): n={n_necessario} | disponivel={N} ✓')

# -----------------------------------------------------------------------
# 3. FORMULA OLS
# -----------------------------------------------------------------------
# Verificar colunas disponíveis
tem_elemento = 'elemento_cat' in df.columns and df['elemento_cat'].nunique() > 1
tem_tipo     = 'tipo_empenho' in df.columns and df['tipo_empenho'].nunique() > 1

formula_partes = ['log_valorEmpenhado', 'n_pagamentos', 'ano_empenho', 'trim_2', 'trim_3', 'trim_4']
if tem_elemento:
    formula_partes.append('C(elemento_cat)')
if tem_tipo:
    formula_partes.append('C(tipo_empenho)')

FORMULA = 'log_target ~ ' + ' + '.join(formula_partes)
print(f'\n=== MODELO OLS ===')
print(f'Formula: {FORMULA}')

# -----------------------------------------------------------------------
# 4. TREINO / TESTE
# -----------------------------------------------------------------------
df_modelo = df.dropna(subset=['log_target', 'log_valorEmpenhado']).copy()
df_treino, df_teste = train_test_split(df_modelo, test_size=0.2, random_state=42)
print(f'Treino: {len(df_treino)} | Teste: {len(df_teste)}')

# -----------------------------------------------------------------------
# 5. AJUSTAR OLS
# -----------------------------------------------------------------------
modelo = smf.ols(FORMULA, data=df_treino).fit()

print('\n' + '='*70)
print(modelo.summary().as_text())
print('='*70)

# Salvar summary
with open('resultados/checkpoint2_ols_summary.txt', 'w', encoding='utf-8') as f:
    f.write(str(modelo.summary()))
print('\nSalvo: resultados/checkpoint2_ols_summary.txt')

# -----------------------------------------------------------------------
# 6. METRICAS NO CONJUNTO DE TESTE
# -----------------------------------------------------------------------
y_pred_log = modelo.predict(df_teste)
y_real_log = df_teste['log_target']

rmse_log = np.sqrt(np.mean((y_real_log - y_pred_log)**2))
mae_log  = np.mean(np.abs(y_real_log - y_pred_log))
r2_treino = modelo.rsquared
r2_teste  = 1 - np.sum((y_real_log - y_pred_log)**2) / np.sum((y_real_log - y_real_log.mean())**2)

print(f'\n=== METRICAS ===')
print(f'  R² treino    : {r2_treino:.4f}')
print(f'  R² teste     : {r2_teste:.4f}')
print(f'  RMSE (log)   : {rmse_log:.4f}')
print(f'  MAE (log)    : {mae_log:.4f}')

# -----------------------------------------------------------------------
# 7. VERIFICACAO DE PRESSUPOSTOS
# -----------------------------------------------------------------------
residuos = modelo.resid
fitted   = modelo.fittedvalues

print(f'\n=== PRESSUPOSTOS DO OLS ===')

# 7.1 Normalidade dos residuos
amostra_res = residuos.sample(min(5000, len(residuos)), random_state=42).values
W_res, p_res = stats.shapiro(amostra_res)
print(f'  Normalidade (Shapiro-Wilk, n={len(amostra_res)}):')
print(f'    W = {W_res:.4f}, p = {p_res:.2e}')
print(f'    {"OK (p>=0.05)" if p_res >= 0.05 else "ATENCAO (p<0.05, mas n grande pode rejeitar H0 com pequena violacao)"}')

# 7.2 Homocedasticidade (Breusch-Pagan)
try:
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(residuos, modelo.model.exog)
    print(f'  Homocedasticidade (Breusch-Pagan):')
    print(f'    LM = {bp_lm:.4f}, p = {bp_p:.4f}')
    print(f'    {"OK" if bp_p >= 0.05 else "ATENCAO: heterocedasticidade detectada"}')
    bp_resultado = {'lm': round(bp_lm, 4), 'p': round(bp_p, 4), 'homoscedastico': bp_p >= 0.05}
except Exception as e:
    print(f'  Breusch-Pagan: nao calculado ({e})')
    bp_resultado = {}

# 7.3 VIF
try:
    X_vif = modelo.model.exog
    vif_vals = [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
    vif_names = modelo.model.exog_names
    vif_df = pd.DataFrame({'feature': vif_names, 'VIF': vif_vals})
    vif_df = vif_df[vif_df['feature'] != 'Intercept'].sort_values('VIF', ascending=False)
    vif_max = vif_df['VIF'].max()
    print(f'  Multicolinearidade (VIF):')
    print(vif_df.to_string(index=False))
    print(f'    VIF max = {vif_max:.2f} {"OK (< 10)" if vif_max < 10 else "PROBLEMA: remover features com VIF>10"}')
    vif_resultado = vif_df.to_dict(orient='records')
except Exception as e:
    print(f'  VIF: nao calculado ({e})')
    vif_resultado = []

# -----------------------------------------------------------------------
# 8. GRAFICOS DE DIAGNOSTICO
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(16, 12))
fig.suptitle('CHECKPOINT 2 — Diagnostico do Modelo OLS', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig)

# Residuos vs Fitted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(fitted, residuos, alpha=0.3, s=10, color='steelblue')
ax1.axhline(0, color='red', lw=1)
ax1.set_xlabel('Valores Ajustados')
ax1.set_ylabel('Residuos')
ax1.set_title('Residuos vs. Ajustados')

# QQ dos residuos
ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(amostra_res, dist='norm', plot=ax2)
ax2.set_title(f'QQ-Plot Residuos\nSW: W={W_res:.4f}, p={p_res:.2e}')

# Histograma dos residuos
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(amostra_res, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
xr = np.linspace(amostra_res.min(), amostra_res.max(), 200)
ax3.plot(xr, stats.norm.pdf(xr, np.mean(amostra_res), np.std(amostra_res)), 'r-', lw=2)
ax3.set_title('Distribuicao dos Residuos')
ax3.set_xlabel('Residuos')

# Previsao vs Real
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_real_log, y_pred_log, alpha=0.2, s=8, color='steelblue')
lim_min = min(y_real_log.min(), y_pred_log.min())
lim_max = max(y_real_log.max(), y_pred_log.max())
ax4.plot([lim_min, lim_max], [lim_min, lim_max], 'r-', lw=1.5, label='Linha perfeita')
ax4.set_xlabel('log_target real')
ax4.set_ylabel('log_target previsto')
ax4.set_title(f'Real vs. Previsto\nR² treino={r2_treino:.3f} | R² teste={r2_teste:.3f}')
ax4.legend(fontsize=8)

# Coeficientes significativos
ax5 = fig.add_subplot(gs[1, 1])
coefs = modelo.params.drop('Intercept', errors='ignore')
pvals = modelo.pvalues.drop('Intercept', errors='ignore')
sig = coefs[pvals < 0.05]
ci = modelo.conf_int().drop('Intercept', errors='ignore')
ci_sig = ci[pvals < 0.05]
erros = (ci_sig[1] - ci_sig[0]) / 2
y_pos = range(len(sig))
colors_coef = ['green' if v > 0 else 'red' for v in sig.values]
ax5.barh(list(y_pos), sig.values, xerr=erros.values, color=colors_coef, alpha=0.7, capsize=3)
ax5.set_yticks(list(y_pos))
ax5.set_yticklabels([n[:20] for n in sig.index], fontsize=7)
ax5.axvline(0, color='black', lw=0.8)
ax5.set_title(f'Coef. Significativos (p<0.05)\n{len(sig)} de {len(coefs)} preditores')

# Tabela de métricas
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
tabela = [
    ['Metrica', 'Valor'],
    ['R² (treino)', f'{r2_treino:.4f}'],
    ['R² (teste)', f'{r2_teste:.4f}'],
    ['RMSE (log)', f'{rmse_log:.4f}'],
    ['MAE (log)', f'{mae_log:.4f}'],
    ['N treino', str(len(df_treino))],
    ['N teste', str(len(df_teste))],
    ['Preditores', str(int(modelo.df_model))],
    ['F-stat p', f'{modelo.f_pvalue:.2e}'],
    ['VIF max', f'{vif_max:.2f}' if vif_resultado else 'n/a'],
]
t = ax6.table(cellText=tabela[1:], colLabels=tabela[0], loc='center', cellLoc='center')
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1.2, 1.4)
ax6.set_title('Resumo do Modelo', fontsize=10, pad=15)

plt.tight_layout()
plt.savefig('graficos/checkpoint2_diagnostico.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSalvo: graficos/checkpoint2_diagnostico.png')

# -----------------------------------------------------------------------
# 9. SALVAR MODELO E RESULTADOS
# -----------------------------------------------------------------------
modelo.save('resultados/checkpoint2_modelo.pkl')
print('Salvo: resultados/checkpoint2_modelo.pkl')

resultado = {
    'tamanho_amostral': resultados_amostra,
    'n_necessario_principal': n_necessario,
    'n_disponivel': N,
    'formula': FORMULA,
    'metricas': {
        'r2_treino': round(r2_treino, 4),
        'r2_teste': round(r2_teste, 4),
        'rmse_log': round(rmse_log, 4),
        'mae_log': round(mae_log, 4),
    },
    'f_statistic': {'value': round(modelo.fvalue, 2), 'p': float(modelo.f_pvalue)},
    'pressupostos': {
        'normalidade_residuos': {'W': round(W_res, 4), 'p': float(p_res), 'ok': bool(p_res >= 0.05)},
        'homocedasticidade': {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in bp_resultado.items()} if bp_resultado else {},
        'vif': vif_resultado,
    }
}
with open('resultados/checkpoint2_metricas.json', 'w', encoding='utf-8') as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)
print('Salvo: resultados/checkpoint2_metricas.json')

print('\n' + '='*50)
print('CHECKPOINT 2 CONCLUIDO')
print('='*50)
print(f'  n necessario   : {n_necessario} | disponivel : {N} ✓')
print(f'  R² treino      : {r2_treino:.4f}')
print(f'  R² teste       : {r2_teste:.4f}')
print(f'  F-stat p-valor : {modelo.f_pvalue:.2e}')
print(f'\nProximos passos:')
print('  python scripts/05_checkpoint3_estimativas.py')
