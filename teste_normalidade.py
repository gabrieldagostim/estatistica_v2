import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, boxcox, yeojohnson, norm, probplot
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("=" * 80)
print("TESTE DE NORMALIDADE - VARIAVEL ALVO (log_target)")
print("=" * 80)
print()

df = pd.read_csv('base_analitica_final.csv')
print(f"[OK] Base carregada: {len(df)} registros x {len(df.columns)} colunas")
print()

# Trabalhar com a variável alvo (log_target)
target = df['log_target'].dropna()
print(f"[OK] Variavel alvo (log_target): {len(target)} valores nao-nulos")
print()

# ============================================================================
# 2. TESTE SHAPIRO-WILK NA SÉRIE ORIGINAL
# ============================================================================
print("-" * 80)
print("TESTE 1: SHAPIRO-WILK NA SERIE ORIGINAL (log_target)")
print("-" * 80)

# Nota: Shapiro-Wilk tem limite de 5000 amostras
if len(target) > 5000:
    sample_size = 5000
    target_sample = target.sample(n=sample_size, random_state=42)
    print(f"[AVISO] Base > 5000 amostras. Testando amostra de {sample_size} registros.")
else:
    target_sample = target

statistic_orig, p_value_orig = shapiro(target_sample)

print(f"Estatistica W: {statistic_orig:.6f}")
print(f"P-valor: {p_value_orig:.2e}")
print()

if p_value_orig > 0.05:
    print("[OK] RESULTADO: A serie e NORMAL (p-value > 0.05)")
    normalidade_original = True
else:
    print("[AVISO] RESULTADO: A serie NAO e NORMAL (p-value < 0.05)")
    print("   -> Aplicando transformacoes para normalizacao...")
    normalidade_original = False

print()

# ============================================================================
# 3. ESTATÍSTICAS DESCRITIVAS
# ============================================================================
print("-" * 80)
print("ESTATISTICAS DESCRITIVAS - SERIE ORIGINAL")
print("-" * 80)
print(f"Media: {target.mean():.4f}")
print(f"Mediana: {target.median():.4f}")
print(f"Desvio Padrao: {target.std():.4f}")
print(f"Minimo: {target.min():.4f}")
print(f"Maximo: {target.max():.4f}")
print(f"Assimetria (Skewness): {target.skew():.4f}")
print(f"Curtose (Kurtosis): {target.kurtosis():.4f}")
print()

# ============================================================================
# 4. TRANSFORMAÇÕES PARA NORMALIZAÇÃO
# ============================================================================
transformacoes = {}

if not normalidade_original:
    print("-" * 80)
    print("TESTANDO TRANSFORMACOES PARA NORMALIZACAO")
    print("-" * 80)
    print()

    # 4.1 - Box-Cox (requer valores > 0)
    if (target > 0).all():
        try:
            transformed_boxcox, lambda_bc = boxcox(target)
            stat_bc, p_bc = shapiro(transformed_boxcox[::11] if len(transformed_boxcox) > 5000
                                    else transformed_boxcox)
            transformacoes['Box-Cox'] = {
                'data': transformed_boxcox,
                'stat': stat_bc,
                'p_value': p_bc,
                'lambda': lambda_bc
            }
            print(f"1. BOX-COX")
            print(f"   lambda: {lambda_bc:.4f}")
            print(f"   Estatistica W: {stat_bc:.6f}")
            print(f"   P-valor: {p_bc:.2e}")
            print(f"   Status: {'[OK] NORMAL' if p_bc > 0.05 else '[AVISO] NAO-NORMAL'}")
            print()
        except Exception as e:
            print(f"1. BOX-COX: Erro - {e}\n")

    # 4.2 - Yeo-Johnson (funciona com qualquer valor)
    try:
        transformed_yj, lambda_yj = yeojohnson(target)
        stat_yj, p_yj = shapiro(transformed_yj[::11] if len(transformed_yj) > 5000
                               else transformed_yj)
        transformacoes['Yeo-Johnson'] = {
            'data': transformed_yj,
            'stat': stat_yj,
            'p_value': p_yj,
            'lambda': lambda_yj
        }
        print(f"2. YEO-JOHNSON")
        print(f"   lambda: {lambda_yj:.4f}")
        print(f"   Estatistica W: {stat_yj:.6f}")
        print(f"   P-valor: {p_yj:.2e}")
        print(f"   Status: {'[OK] NORMAL' if p_yj > 0.05 else '[AVISO] NAO-NORMAL'}")
        print()
    except Exception as e:
        print(f"2. YEO-JOHNSON: Erro - {e}\n")

    # 4.3 - Log (se já não é log_target)
    if (target > 0).all():
        try:
            transformed_log = np.log(target)
            stat_log, p_log = shapiro(transformed_log[::11] if len(transformed_log) > 5000
                                     else transformed_log)
            transformacoes['Log'] = {
                'data': transformed_log,
                'stat': stat_log,
                'p_value': p_log,
                'lambda': None
            }
            print(f"3. LOG NATURAL")
            print(f"   Estatistica W: {stat_log:.6f}")
            print(f"   P-valor: {p_log:.2e}")
            print(f"   Status: {'[OK] NORMAL' if p_log > 0.05 else '[AVISO] NAO-NORMAL'}")
            print()
        except Exception as e:
            print(f"3. LOG NATURAL: Erro - {e}\n")

    # 4.4 - Raiz quadrada
    if (target >= 0).all():
        try:
            transformed_sqrt = np.sqrt(target)
            stat_sqrt, p_sqrt = shapiro(transformed_sqrt[::11] if len(transformed_sqrt) > 5000
                                       else transformed_sqrt)
            transformacoes['Raiz Quadrada'] = {
                'data': transformed_sqrt,
                'stat': stat_sqrt,
                'p_value': p_sqrt,
                'lambda': None
            }
            print(f"4. RAIZ QUADRADA")
            print(f"   Estatistica W: {stat_sqrt:.6f}")
            print(f"   P-valor: {p_sqrt:.2e}")
            print(f"   Status: {'[OK] NORMAL' if p_sqrt > 0.05 else '[AVISO] NAO-NORMAL'}")
            print()
        except Exception as e:
            print(f"4. RAIZ QUADRADA: Erro - {e}\n")

    # 4.5 - Inversa (1/x)
    if (target > 0).all():
        try:
            transformed_inv = 1 / target
            stat_inv, p_inv = shapiro(transformed_inv[::11] if len(transformed_inv) > 5000
                                     else transformed_inv)
            transformacoes['Inversa'] = {
                'data': transformed_inv,
                'stat': stat_inv,
                'p_value': p_inv,
                'lambda': None
            }
            print(f"5. INVERSA (1/x)")
            print(f"   Estatistica W: {stat_inv:.6f}")
            print(f"   P-valor: {p_inv:.2e}")
            print(f"   Status: {'[OK] NORMAL' if p_inv > 0.05 else '[AVISO] NAO-NORMAL'}")
            print()
        except Exception as e:
            print(f"5. INVERSA: Erro - {e}\n")

    # 4.6 - Padronização (Z-score)
    try:
        mean_t = target.mean()
        std_t = target.std()
        transformed_zscore = (target - mean_t) / std_t
        stat_z, p_z = shapiro(transformed_zscore[::11] if len(transformed_zscore) > 5000
                             else transformed_zscore)
        transformacoes['Z-Score'] = {
            'data': transformed_zscore,
            'stat': stat_z,
            'p_value': p_z,
            'lambda': None
        }
        print(f"6. PADRONIZACAO (Z-SCORE)")
        print(f"   Estatistica W: {stat_z:.6f}")
        print(f"   P-valor: {p_z:.2e}")
        print(f"   Status: {'[OK] NORMAL' if p_z > 0.05 else '[AVISO] NAO-NORMAL'}")
        print()
    except Exception as e:
        print(f"6. PADRONIZACAO: Erro - {e}\n")

    # ============================================================================
    # 5. RESUMO COMPARATIVO
    # ============================================================================
    print("-" * 80)
    print("RESUMO COMPARATIVO")
    print("-" * 80)

    resultados = []
    resultados.append(['Original (log_target)', statistic_orig, p_value_orig,
                      'Normal' if p_value_orig > 0.05 else 'Nao-Normal'])

    for nome, dados in transformacoes.items():
        resultados.append([nome, dados['stat'], dados['p_value'],
                          'Normal' if dados['p_value'] > 0.05 else 'Nao-Normal'])

    df_resultado = pd.DataFrame(resultados, columns=['Transformacao', 'Estatistica W', 'P-valor', 'Resultado'])
    df_resultado = df_resultado.sort_values('P-valor', ascending=False)
    print(df_resultado.to_string(index=False))
    print()

    # Melhor transformação
    best_transform = df_resultado.iloc[0]
    print(f"[MELHOR] Transformacao: {best_transform['Transformacao']}")
    print(f"   P-valor: {best_transform['P-valor']:.2e}")
    print()

# ============================================================================
# 6. VISUALIZAÇÕES
# ============================================================================
print("-" * 80)
print("GERANDO VISUALIZACOES...")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Teste de Normalidade - Variavel Alvo (log_target)', fontsize=14, fontweight='bold')

# Original
ax = axes[0, 0]
ax.hist(target, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax.set_title(f'Original\nW={statistic_orig:.4f}, p={p_value_orig:.2e}')
ax.set_xlabel('log_target')
ax.set_ylabel('Frequencia')

# Q-Q Plot Original
ax = axes[0, 1]
probplot(target, dist="norm", plot=ax)
ax.set_title('Q-Q Plot - Original')

# Distribuição Original com Normal overlay
ax = axes[0, 2]
mu, sigma = target.mean(), target.std()
x = np.linspace(target.min(), target.max(), 100)
ax.hist(target, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Teorica')
ax.set_title('Original vs Distribuicao Normal')
ax.set_xlabel('log_target')
ax.set_ylabel('Densidade')
ax.legend()

# Se houver transformações, plotar as 3 melhores
if not normalidade_original and transformacoes:
    sorted_transforms = sorted(transformacoes.items(),
                              key=lambda x: x[1]['p_value'], reverse=True)

    for idx, (nome, dados) in enumerate(sorted_transforms[:3]):
        row = 1
        col = idx
        if row < 2 and col < 3:
            ax = axes[row, col]
            transformed_data = dados['data']

            ax.hist(transformed_data, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
            ax.set_title(f'{nome}\nW={dados["stat"]:.4f}, p={dados["p_value"]:.2e}')
            ax.set_xlabel(f'{nome} (transformado)')
            ax.set_ylabel('Frequencia')

plt.tight_layout()
plt.savefig('teste_normalidade_visualizacao.png', dpi=300, bbox_inches='tight')
print("[OK] Grafico salvo: teste_normalidade_visualizacao.png")
print()

# ============================================================================
# 7. SALVAR DADOS TRANSFORMADOS
# ============================================================================
print("-" * 80)
print("SALVANDO DADOS TRANSFORMADOS...")
print("-" * 80)

df_transformados = df.copy()

# Adicionar transformações
if not normalidade_original and transformacoes:
    for nome, dados in transformacoes.items():
        col_name = f'target_{nome.lower().replace("-", "_").replace(" ", "_")}'
        df_transformados[col_name] = dados['data']
        print(f"[OK] Coluna adicionada: {col_name}")

df_transformados.to_csv('base_analitica_com_transformacoes.csv', index=False)
print(f"[OK] Arquivo salvo: base_analitica_com_transformacoes.csv")
print()

# ============================================================================
# 8. RELATÓRIO FINAL
# ============================================================================
print("=" * 80)
print("CONCLUSAO")
print("=" * 80)

if normalidade_original:
    print("[OK] A serie ORIGINAL ja segue uma distribuicao NORMAL!")
    print(f"   P-valor do teste Shapiro-Wilk: {p_value_orig:.2e}")
    print()
    print("Recomendacoes:")
    print("  • Usar a serie original (log_target) para modelagem linear")
    print("  • Metodos parametricos (ANOVA, regressao linear) sao apropriados")
else:
    print("[AVISO] A serie original NAO segue distribuicao normal")
    print()
    print("Recomendacoes:")
    print("  • Usar transformacoes para modelagem linear (veja resumo acima)")
    print("  • Considerar modelos nao-parametricos ou robustos (XGBoost, LightGBM)")
    print("  • As transformacoes testadas foram salvas no arquivo de saida")

print()
print("=" * 80)
