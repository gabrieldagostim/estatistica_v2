"""
CHECKPOINT 3 — Estimativas Pontuais e Intervalares + Previsoes OLS
Trabalho de Extensao — Estatistica Aplicada — Transparencia Criciuma/SC
"""
import os, json, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.iolib.smpickle import load_pickle
warnings.filterwarnings('ignore')

os.makedirs('graficos', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

# -----------------------------------------------------------------------
# 1. CARREGAR BASE E MODELO
# -----------------------------------------------------------------------
BASE   = 'bases/base_ols.csv'
MODELO = 'resultados/checkpoint2_modelo.pkl'

if not os.path.exists(BASE):
    print(f'ERRO: {BASE} nao encontrado. Execute scripts/02_features_ols.py')
    exit(1)
if not os.path.exists(MODELO):
    print(f'ERRO: {MODELO} nao encontrado. Execute scripts/04_checkpoint2_amostragem_ols.py')
    exit(1)

df = pd.read_csv(BASE)
modelo = load_pickle(MODELO)

N = len(df)
print(f'Base: {N} registros | Modelo OLS carregado')

# -----------------------------------------------------------------------
# 2. ESTIMATIVAS PONTUAIS — variavel-alvo original
# -----------------------------------------------------------------------
print('\n=== ESTIMATIVAS PONTUAIS ===')

y_orig = df['valorPagoEmpenho'].dropna()
y_log  = df['log_target'].dropna()

# Media, mediana, desvio
media_orig   = y_orig.mean()
mediana_orig = y_orig.median()
dp_orig      = y_orig.std(ddof=1)
ep_media     = dp_orig / math.sqrt(len(y_orig))  # erro padrao da media

media_log    = y_log.mean()
mediana_log  = y_log.median()
dp_log       = y_log.std(ddof=1)
ep_media_log = dp_log / math.sqrt(len(y_log))

print(f'\n  valorPagoEmpenho (escala original):')
print(f'    Media      : R$ {media_orig:,.2f}')
print(f'    Mediana    : R$ {mediana_orig:,.2f}')
print(f'    Desvio Pad.: R$ {dp_orig:,.2f}')
print(f'    Erro Padrao: R$ {ep_media:,.2f}')

print(f'\n  log_target (escala log):')
print(f'    Media      : {media_log:.4f}')
print(f'    Mediana    : {mediana_log:.4f}')
print(f'    Desvio Pad.: {dp_log:.4f}')
print(f'    Erro Padrao: {ep_media_log:.6f}')

# -----------------------------------------------------------------------
# 3. FUNCAO IC PARA A MEDIA
# -----------------------------------------------------------------------
def ic_media(dados, confianca=0.95, populacao_finita=False, N_pop=None):
    n   = len(dados)
    m   = np.mean(dados)
    s   = np.std(dados, ddof=1)
    ep  = s / math.sqrt(n)
    alpha = 1 - confianca
    if n < 30:
        tc = stats.t.ppf(1 - alpha/2, df=n-1)
        metodo = 't-Student'
    else:
        tc = stats.norm.ppf(1 - alpha/2)
        metodo = 'Normal (z)'
    fpc = 1.0
    if populacao_finita and N_pop:
        fpc = math.sqrt((N_pop - n) / (N_pop - 1))
        ep *= fpc
    margem = tc * ep
    return {
        'metodo': metodo, 'n': n, 'media': m, 'ep': ep,
        'tc': tc, 'fpc': fpc, 'margem': margem,
        'li': m - margem, 'ls': m + margem,
        'confianca': confianca
    }

# -----------------------------------------------------------------------
# 4. ESTIMATIVAS INTERVALARES
# -----------------------------------------------------------------------
print('\n=== INTERVALOS DE CONFIANCA ===')

# IC para a media do log_target (populacao infinita)
ic_log = ic_media(y_log.values, confianca=0.95)
print(f'\nIC 95% para media de log_target (n={ic_log["n"]}):')
print(f'  Metodo  : {ic_log["metodo"]}')
print(f'  Media   : {ic_log["media"]:.4f}')
print(f'  Erro P. : {ic_log["ep"]:.6f}')
print(f'  Margem  : ±{ic_log["margem"]:.6f}')
print(f'  IC 95%  : [{ic_log["li"]:.4f}, {ic_log["ls"]:.4f}]')

# IC para a media do valor original
ic_orig = ic_media(y_orig.values, confianca=0.95)
print(f'\nIC 95% para media de valorPagoEmpenho (R$):')
print(f'  Media   : R$ {ic_orig["media"]:,.2f}')
print(f'  Margem  : ±R$ {ic_orig["margem"]:,.2f}')
print(f'  IC 95%  : [R$ {ic_orig["li"]:,.2f}, R$ {ic_orig["ls"]:,.2f}]')

# IC para a media — com correcao de populacao finita
ic_finita = ic_media(y_log.values, confianca=0.95, populacao_finita=True, N_pop=N)
print(f'\nIC 95% para media (c/ correcao finita, N={N}):')
print(f'  FPC     : {ic_finita["fpc"]:.6f}')
print(f'  IC 95%  : [{ic_finita["li"]:.4f}, {ic_finita["ls"]:.4f}]')

# IC para proporcao — empenhos acima de R$ 5.000
limiar = 5000
p_hat = (y_orig > limiar).mean()
n_hat = len(y_orig)
ep_prop = math.sqrt(p_hat * (1 - p_hat) / n_hat)
z95 = stats.norm.ppf(0.975)
print(f'\nIC 95% para proporcao de empenhos > R$ {limiar:,.0f}:')
print(f'  p_hat   : {p_hat:.4f} ({p_hat*100:.2f}%)')
print(f'  EP prop : {ep_prop:.4f}')
print(f'  Margem  : ±{z95*ep_prop:.4f}')
print(f'  IC 95%  : [{(p_hat - z95*ep_prop)*100:.2f}%, {(p_hat + z95*ep_prop)*100:.2f}%]')

# Diferenca de medias entre dois elementos (se disponivel)
ic_diff = None
if 'elemento_cat' in df.columns:
    elementos = df['elemento_cat'].value_counts().head(2).index.tolist()
    if len(elementos) >= 2:
        g1 = df[df['elemento_cat'] == elementos[0]]['log_target'].dropna().values
        g2 = df[df['elemento_cat'] == elementos[1]]['log_target'].dropna().values
        t_stat, p_ttest = stats.ttest_ind(g1, g2)
        diff = np.mean(g1) - np.mean(g2)
        ep_diff = math.sqrt(np.var(g1, ddof=1)/len(g1) + np.var(g2, ddof=1)/len(g2))
        margem_diff = z95 * ep_diff
        print(f'\nIC 95% para diferenca de medias:')
        print(f'  {elementos[0][:25]} vs {elementos[1][:25]}')
        print(f'  Diferenca  : {diff:.4f}')
        print(f'  Margem     : ±{margem_diff:.4f}')
        print(f'  IC 95%     : [{diff-margem_diff:.4f}, {diff+margem_diff:.4f}]')
        print(f'  t-test p   : {p_ttest:.2e} -> {"Signif." if p_ttest < 0.05 else "Nao signif."}')
        ic_diff = {'elementos': elementos, 'diferenca': round(diff, 4),
                   'li': round(diff-margem_diff, 4), 'ls': round(diff+margem_diff, 4),
                   'p_ttest': float(p_ttest)}

# -----------------------------------------------------------------------
# 5. PREVISOES OLS COM INTERVALO DE PREDICAO
# -----------------------------------------------------------------------
print('\n=== PREVISOES COM INTERVALO DE PREDICAO (95%) ===')

# Construir cenarios futuros
if 'elemento_cat' in df.columns:
    top_elem = df['elemento_cat'].value_counts().head(3).index.tolist()
else:
    top_elem = ['Vencimentos']

if 'tipo_empenho' in df.columns:
    tipo_mais_comum = df['tipo_empenho'].value_counts().index[0]
else:
    tipo_mais_comum = 'Ordinario'

ano_futuro = df['ano_empenho'].max() + 1 if 'ano_empenho' in df.columns else 2027

log_emp_mediano = df['log_valorEmpenhado'].median()
n_pag_medio     = df['n_pagamentos'].median()

cenarios = []
for elem in top_elem[:3]:
    for trim, t2, t3, t4 in [(1, 0, 0, 0), (2, 1, 0, 0), (3, 0, 1, 0)]:
        cenarios.append({
            'elemento_cat': elem,
            'tipo_empenho': tipo_mais_comum,
            'log_valorEmpenhado': log_emp_mediano,
            'n_pagamentos': n_pag_medio,
            'ano_empenho': ano_futuro,
            'mes_empenho': trim * 3,
            'trimestre': trim,
            'trim_2': t2,
            'trim_3': t3,
            'trim_4': t4,
        })

df_cenarios = pd.DataFrame(cenarios)

try:
    predicoes = modelo.get_prediction(df_cenarios)
    pred_frame = predicoes.summary_frame(alpha=0.05)

    print(f'\nPrevisoes para ano {ano_futuro}:')
    print(f'{"Elemento":<30} {"Trim":<5} {"Prev.(log)":<12} {"Real(R$)":<12} {"IC_Pred [Li, Ls]"}')
    print('-' * 90)

    resultados_prev = []
    for i, (_, row) in enumerate(df_cenarios.iterrows()):
        pf = pred_frame.iloc[i]
        mean_log = pf['mean']
        li_log   = pf['obs_ci_lower']
        ls_log   = pf['obs_ci_upper']
        real     = math.expm1(mean_log)
        li_real  = math.expm1(li_log)
        ls_real  = math.expm1(ls_log)
        elem     = row['elemento_cat'][:28]
        trim     = int(row['trimestre'])
        print(f'{elem:<30} Q{trim:<4} {mean_log:<12.4f} R${real:<11,.0f} [R${li_real:,.0f}, R${ls_real:,.0f}]')
        resultados_prev.append({
            'elemento': row['elemento_cat'], 'trimestre': trim, 'ano': ano_futuro,
            'previsao_log': round(mean_log, 4), 'previsao_real': round(real, 2),
            'ic_pred_li': round(li_real, 2), 'ic_pred_ls': round(ls_real, 2)
        })
except Exception as e:
    print(f'Erro ao calcular previsoes: {e}')
    resultados_prev = []

# -----------------------------------------------------------------------
# 6. GRAFICOS
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CHECKPOINT 3 — Estimativas e Previsoes', fontsize=13, fontweight='bold')

# Histograma com IC
ax = axes[0, 0]
ax.hist(y_log.values, bins=60, density=True, alpha=0.6, color='steelblue', edgecolor='white')
ax.axvline(media_log, color='red', lw=2, label=f'Media={media_log:.3f}')
ax.axvline(ic_log['li'], color='orange', lw=1.5, ls='--', label=f'IC 95%: [{ic_log["li"]:.3f}, {ic_log["ls"]:.3f}]')
ax.axvline(ic_log['ls'], color='orange', lw=1.5, ls='--')
ax.set_title('Distribuicao log_target com IC para a Media')
ax.set_xlabel('log_target')
ax.legend(fontsize=8)

# IC por elemento
ax = axes[0, 1]
if 'elemento_cat' in df.columns:
    top_elem_ic = df['elemento_cat'].value_counts().head(6).index.tolist()
    medias = []
    erros  = []
    for e in top_elem_ic:
        g = df[df['elemento_cat'] == e]['log_target'].dropna().values
        ic_e = ic_media(g, confianca=0.95)
        medias.append(ic_e['media'])
        erros.append(ic_e['margem'])
    ax.barh(range(len(top_elem_ic)), medias, xerr=erros, color='steelblue', alpha=0.7, capsize=4)
    ax.set_yticks(range(len(top_elem_ic)))
    ax.set_yticklabels([e[:25] for e in top_elem_ic], fontsize=8)
    ax.set_title('IC 95% para Media por Elemento')
    ax.set_xlabel('log_target (media ± IC)')
else:
    ax.text(0.5, 0.5, 'elemento_cat nao disponivel', ha='center', va='center', transform=ax.transAxes)

# Previsoes com IC
ax = axes[1, 0]
if resultados_prev:
    df_prev = pd.DataFrame(resultados_prev)
    labels = [f"{r['elemento'][:18]}\nQ{r['trimestre']}" for _, r in df_prev.iterrows()]
    yerr_low = df_prev['previsao_real'] - df_prev['ic_pred_li']
    yerr_high = df_prev['ic_pred_ls'] - df_prev['previsao_real']
    x_pos = range(len(df_prev))
    ax.errorbar(x_pos, df_prev['previsao_real'], yerr=[yerr_low, yerr_high],
                fmt='o', capsize=5, color='coral', markersize=6)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
    ax.set_title(f'Previsoes para {ano_futuro} com IC Pred. 95%')
    ax.set_ylabel('valorPagoEmpenho previsto (R$)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'R${x:,.0f}'))
else:
    ax.text(0.5, 0.5, 'Previsoes nao calculadas', ha='center', va='center', transform=ax.transAxes)

# Tabela de estimativas
ax = axes[1, 1]
ax.axis('off')
tabela = [
    ['Estimativa', 'Valor'],
    ['Media original', f'R$ {media_orig:,.2f}'],
    ['Mediana original', f'R$ {mediana_orig:,.2f}'],
    ['Desvio padrao', f'R$ {dp_orig:,.2f}'],
    ['EP da media', f'R$ {ep_media:,.2f}'],
    ['IC 95% media (log)', f'[{ic_log["li"]:.4f}, {ic_log["ls"]:.4f}]'],
    ['IC 95% media (R$)', f'[{ic_orig["li"]:,.0f}, {ic_orig["ls"]:,.0f}]'],
    [f'Prop. > R${limiar:,}', f'{p_hat*100:.1f}% [{(p_hat-z95*ep_prop)*100:.1f}%, {(p_hat+z95*ep_prop)*100:.1f}%]'],
]
t = ax.table(cellText=tabela[1:], colLabels=tabela[0], loc='center', cellLoc='center')
t.auto_set_font_size(False)
t.set_fontsize(8)
t.scale(1.2, 1.5)
ax.set_title('Resumo das Estimativas', fontsize=10, pad=15)

plt.tight_layout()
plt.savefig('graficos/checkpoint3_estimativas.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSalvo: graficos/checkpoint3_estimativas.png')

# -----------------------------------------------------------------------
# 7. SALVAR RESULTADOS
# -----------------------------------------------------------------------
resultado = {
    'estimativas_pontuais': {
        'media_original': round(media_orig, 2),
        'mediana_original': round(mediana_orig, 2),
        'desvio_padrao': round(dp_orig, 2),
        'erro_padrao_media': round(ep_media, 2),
        'media_log': round(media_log, 4),
        'dp_log': round(dp_log, 4),
    },
    'intervalos_confianca': {
        'ic_media_log_95': {'li': round(ic_log['li'], 4), 'ls': round(ic_log['ls'], 4)},
        'ic_media_real_95': {'li': round(ic_orig['li'], 2), 'ls': round(ic_orig['ls'], 2)},
        'ic_media_finita_95': {'li': round(ic_finita['li'], 4), 'ls': round(ic_finita['ls'], 4)},
        'ic_proporcao_95': {
            'proporcao': round(p_hat, 4),
            'limiar': limiar,
            'li_pct': round((p_hat - z95*ep_prop)*100, 2),
            'ls_pct': round((p_hat + z95*ep_prop)*100, 2)
        },
        'ic_diferenca_medias': ic_diff,
    },
    'previsoes': resultados_prev,
}
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (bool,)):
            return bool(obj)
        return super().default(obj)

with open('resultados/checkpoint3_estimativas.json', 'w', encoding='utf-8') as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
print('Salvo: resultados/checkpoint3_estimativas.json')

print('\n' + '='*50)
print('CHECKPOINT 3 CONCLUIDO')
print('='*50)
print(f'  Media valorPago : R$ {media_orig:,.2f}')
print(f'  IC 95% (R$)     : [R$ {ic_orig["li"]:,.2f}, R$ {ic_orig["ls"]:,.2f}]')
print(f'  Prop > R$5k     : {p_hat*100:.1f}%')
print(f'  Previsoes geradas: {len(resultados_prev)} cenarios')
print(f'\nTodos os checkpoints concluidos!')
print('Graficos em: graficos/')
print('Resultados em: resultados/')
