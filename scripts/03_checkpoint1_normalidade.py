"""
CHECKPOINT 1 — Teste de Normalidade e Transformacao Box-Cox
Trabalho de Extensao — Estatistica Aplicada — Transparencia Criciuma/SC
"""
import os, json, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, boxcox, probplot
warnings.filterwarnings('ignore')

os.makedirs('graficos', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

# -----------------------------------------------------------------------
# 1. CARREGAR BASE
# -----------------------------------------------------------------------
BASE = 'bases/base_ols.csv'
if not os.path.exists(BASE):
    print(f'ERRO: {BASE} nao encontrado. Execute primeiro:')
    print('  python scripts/02_features_ols.py')
    exit(1)

df = pd.read_csv(BASE)
print(f'Base carregada: {df.shape[0]} registros')

target_original = df['log_target'].dropna().values
N = len(target_original)
print(f'Variavel-alvo: log_target = log1p(valorPagoEmpenho)')
print(f'N = {N}')

# -----------------------------------------------------------------------
# 2. ESTATISTICAS DESCRITIVAS
# -----------------------------------------------------------------------
media      = np.mean(target_original)
mediana    = np.median(target_original)
desvio     = np.std(target_original, ddof=1)
assimetria = stats.skew(target_original)
curtose    = stats.kurtosis(target_original)
cv         = (desvio / media) * 100 if media != 0 else np.nan

print('\n=== ESTATISTICAS DESCRITIVAS (log_target) ===')
print(f'  Media       : {media:.4f}')
print(f'  Mediana     : {mediana:.4f}')
print(f'  Desvio Pad. : {desvio:.4f}')
print(f'  Assimetria  : {assimetria:.4f}')
print(f'  Curtose     : {curtose:.4f}')
print(f'  CV (%)      : {cv:.2f}%')

# -----------------------------------------------------------------------
# 3. SHAPIRO-WILK (ORIGINAL)
# -----------------------------------------------------------------------
AMOSTRA_SW = min(5000, N)
np.random.seed(42)
amostra_idx = np.random.choice(N, AMOSTRA_SW, replace=False)
amostra_orig = target_original[amostra_idx]

W_orig, p_orig = shapiro(amostra_orig)
print(f'\n=== SHAPIRO-WILK (original, n={AMOSTRA_SW}) ===')
print(f'  W = {W_orig:.6f}')
print(f'  p = {p_orig:.2e}')
if p_orig < 0.05:
    print('  >> NAO normal (p < 0.05) -> Box-Cox necessario')
else:
    print('  >> Normal (p >= 0.05) -> transformacao opcional')

# -----------------------------------------------------------------------
# 4. BOX-COX
# -----------------------------------------------------------------------
vals_positivos = target_original[target_original > 0]
if len(vals_positivos) < N:
    print(f'\nAviso: {N - len(vals_positivos)} valores <= 0 excluidos do Box-Cox')

target_bc, lambda_otimo = boxcox(vals_positivos)
print(f'\n=== BOX-COX ===')
print(f'  Lambda otimo : {lambda_otimo:.4f}')

amostra_bc = target_bc[amostra_idx[:len(vals_positivos)]] if len(vals_positivos) == N else target_bc[:AMOSTRA_SW]
amostra_bc = amostra_bc[:AMOSTRA_SW]
W_bc, p_bc = shapiro(amostra_bc)
print(f'  Shapiro-Wilk apos Box-Cox (n={len(amostra_bc)}):')
print(f'    W = {W_bc:.6f}')
print(f'    p = {p_bc:.2e}')

melhoria_p   = (p_bc / p_orig) if p_orig > 0 else np.nan
melhoria_sk  = (1 - abs(stats.skew(target_bc)) / (abs(assimetria) + 1e-9)) * 100
melhoria_kur = (1 - abs(stats.kurtosis(target_bc)) / (abs(curtose) + 1e-9)) * 100

print(f'  Melhoria no p-valor : {melhoria_p:.2f}x')
print(f'  Reducao assimetria  : {melhoria_sk:.1f}%')
print(f'  Reducao curtose     : {melhoria_kur:.1f}%')

# -----------------------------------------------------------------------
# 5. GRAFICOS
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('CHECKPOINT 1 — Normalidade e Transformacao Box-Cox', fontsize=13, fontweight='bold')

# Histograma original
ax = axes[0, 0]
ax.hist(amostra_orig, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 200)
ax.plot(x, stats.norm.pdf(x, np.mean(amostra_orig), np.std(amostra_orig)), 'r-', lw=2, label='Normal teorica')
ax.set_title('Histograma Original (log_target)')
ax.set_xlabel('log_target')
ax.legend(fontsize=8)

# QQ original
ax = axes[0, 1]
probplot(amostra_orig, dist='norm', plot=ax)
ax.set_title(f'QQ-Plot Original\nSW: W={W_orig:.4f}, p={p_orig:.2e}')

# Boxplot original
ax = axes[0, 2]
ax.boxplot(amostra_orig, vert=True, patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.7))
ax.set_title('Boxplot Original')
ax.set_ylabel('log_target')

# Histograma Box-Cox
ax = axes[1, 0]
ax.hist(amostra_bc, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 200)
ax.plot(x, stats.norm.pdf(x, np.mean(amostra_bc), np.std(amostra_bc)), 'r-', lw=2, label='Normal teorica')
ax.set_title(f'Histograma Box-Cox (λ={lambda_otimo:.4f})')
ax.set_xlabel('log_target_bc')
ax.legend(fontsize=8)

# QQ Box-Cox
ax = axes[1, 1]
probplot(amostra_bc, dist='norm', plot=ax)
ax.set_title(f'QQ-Plot Box-Cox\nSW: W={W_bc:.4f}, p={p_bc:.2e}')

# Tabela comparativa
ax = axes[1, 2]
ax.axis('off')
tabela = [
    ['Metrica', 'Original', 'Box-Cox'],
    ['W (Shapiro)', f'{W_orig:.4f}', f'{W_bc:.4f}'],
    ['p-valor', f'{p_orig:.2e}', f'{p_bc:.2e}'],
    ['Assimetria', f'{assimetria:.4f}', f'{stats.skew(target_bc):.4f}'],
    ['Curtose', f'{curtose:.4f}', f'{stats.kurtosis(target_bc):.4f}'],
    ['Lambda', '—', f'{lambda_otimo:.4f}'],
    ['Normal?', 'NAO' if p_orig < 0.05 else 'SIM',
               'NAO (n grande)' if p_bc < 0.05 else 'SIM'],
]
t = ax.table(cellText=tabela[1:], colLabels=tabela[0], loc='center', cellLoc='center')
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1.2, 1.5)
ax.set_title('Comparativo Original vs. Box-Cox', fontsize=10, pad=15)

plt.tight_layout()
plt.savefig('graficos/checkpoint1_normalidade.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSalvo: graficos/checkpoint1_normalidade.png')

# -----------------------------------------------------------------------
# 6. ADICIONAR COLUNA BOX-COX NA BASE
# -----------------------------------------------------------------------
if len(vals_positivos) == N:
    df['log_target_bc'] = target_bc
    df['lambda_boxcox'] = lambda_otimo
    df.to_csv('bases/base_ols.csv', index=False)
    print('Atualizado: bases/base_ols.csv (coluna log_target_bc adicionada)')

# -----------------------------------------------------------------------
# 7. SALVAR RESULTADO
# -----------------------------------------------------------------------
resultado = {
    'n_registros': N,
    'variavel_alvo': 'log_target = log1p(valorPagoEmpenho)',
    'n_candidatas': 25,
    'n_selecionadas_acima_03': 'ver resultados/candidatas_variaveis.csv',
    'shapiro_wilk_original': {'W': round(W_orig, 6), 'p': float(p_orig), 'normal': bool(p_orig >= 0.05)},
    'boxcox': {
        'lambda_otimo': round(lambda_otimo, 4),
        'shapiro_wilk_pos_bc': {'W': round(W_bc, 6), 'p': float(p_bc)},
        'melhoria_p_valor': round(melhoria_p, 2),
        'reducao_assimetria_pct': round(melhoria_sk, 1),
        'reducao_curtose_pct': round(melhoria_kur, 1),
    },
    'estatisticas_originais': {
        'media': round(media, 4),
        'mediana': round(mediana, 4),
        'desvio_padrao': round(desvio, 4),
        'assimetria': round(assimetria, 4),
        'curtose': round(curtose, 4),
        'cv_pct': round(cv, 2),
    }
}
with open('resultados/checkpoint1_resultado.json', 'w', encoding='utf-8') as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)
print('Salvo: resultados/checkpoint1_resultado.json')

print('\n' + '='*50)
print('CHECKPOINT 1 CONCLUIDO')
print('='*50)
print(f'  Registros        : {N}')
print(f'  SW original      : W={W_orig:.4f}, p={p_orig:.2e} -> {"NAO normal" if p_orig < 0.05 else "Normal"}')
print(f'  Box-Cox lambda   : {lambda_otimo:.4f}')
print(f'  SW apos Box-Cox  : W={W_bc:.4f}, p={p_bc:.2e}')
print(f'  Melhoria p-valor : {melhoria_p:.2f}x')
print(f'\nProximos passos:')
print('  python scripts/04_checkpoint2_amostragem_ols.py')
