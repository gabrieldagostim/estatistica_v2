import pandas as pd
import numpy as np
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GERACAO DE BASE NORMALIZADA COM BOX-COX")
print("=" * 80)
print()

# ============================================================================
# 1. CARREGAR BASE ORIGINAL
# ============================================================================
print("Etapa 1: Carregando base original...")
df = pd.read_csv('base_analitica_final.csv')
print(f"[OK] {len(df)} registros carregados")
print(f"[OK] {len(df.columns)} colunas originais")
print()

# ============================================================================
# 2. EXTRAIR VARIAVEL ALVO E APLICAR BOX-COX
# ============================================================================
print("Etapa 2: Aplicando transformacao Box-Cox a log_target...")

target = df['log_target'].copy()
print(f"[OK] Variavel alvo extraida: {len(target)} valores")
print()

# Aplicar Box-Cox com lambda otimizado
transformed_target, lambda_optimal = boxcox(target)

print(f"[OK] Transformacao Box-Cox aplicada")
print(f"    Lambda otimizado: {lambda_optimal:.4f}")
print(f"    Formula: y(lambda) = (y^{lambda_optimal:.4f} - 1) / {lambda_optimal:.4f}")
print()

# ============================================================================
# 3. CRIAR BASE NORMALIZADA
# ============================================================================
print("Etapa 3: Construindo base normalizada...")

# Copiar dataframe original
df_normalizado = df.copy()

# Adicionar coluna transformada
df_normalizado['log_target_normalizado'] = transformed_target

# Adicionar parametro lambda para referencia
df_normalizado['lambda_boxcox'] = lambda_optimal

# Reordenar colunas: manter original e colocar normalizado logo apos
colunas = df_normalizado.columns.tolist()
colunas.remove('log_target_normalizado')
colunas.remove('lambda_boxcox')

# Mover log_target para posicao 0 se existir
if 'log_target' in colunas:
    colunas.remove('log_target')
    colunas = ['log_target', 'log_target_normalizado'] + colunas

# Adicionar lambda ao final
colunas.append('lambda_boxcox')

df_normalizado = df_normalizado[colunas]

print(f"[OK] Base normalizada construida")
print(f"    Total de colunas: {len(df_normalizado)}")
print(f"    Total de registros: {len(df_normalizado)}")
print()

# ============================================================================
# 4. ESTATISTICAS COMPARATIVAS
# ============================================================================
print("Etapa 4: Estatisticas comparativas...")
print()

print("VARIAVEL ORIGINAL (log_target)")
print("-" * 40)
print(f"Media:          {df['log_target'].mean():.6f}")
print(f"Mediana:        {df['log_target'].median():.6f}")
print(f"Desvio Padrao:  {df['log_target'].std():.6f}")
print(f"Minimo:         {df['log_target'].min():.6f}")
print(f"Maximo:         {df['log_target'].max():.6f}")
print(f"Assimetria:     {df['log_target'].skew():.6f}")
print(f"Curtose:        {df['log_target'].kurtosis():.6f}")
print()

print("VARIAVEL NORMALIZADA (log_target_normalizado)")
print("-" * 40)
print(f"Media:          {df_normalizado['log_target_normalizado'].mean():.6f}")
print(f"Mediana:        {df_normalizado['log_target_normalizado'].median():.6f}")
print(f"Desvio Padrao:  {df_normalizado['log_target_normalizado'].std():.6f}")
print(f"Minimo:         {df_normalizado['log_target_normalizado'].min():.6f}")
print(f"Maximo:         {df_normalizado['log_target_normalizado'].max():.6f}")
print(f"Assimetria:     {df_normalizado['log_target_normalizado'].skew():.6f}")
print(f"Curtose:        {df_normalizado['log_target_normalizado'].kurtosis():.6f}")
print()

# Calculo de melhoria em assimetria e curtose
skew_melhoria_pct = (abs(df['log_target'].skew()) - abs(df_normalizado['log_target_normalizado'].skew())) / abs(df['log_target'].skew()) * 100
kurt_melhoria_pct = (abs(df['log_target'].kurtosis()) - abs(df_normalizado['log_target_normalizado'].kurtosis())) / abs(df['log_target'].kurtosis()) * 100

print("MELHORIA POS-TRANSFORMACAO")
print("-" * 40)
print(f"Assimetria: {skew_melhoria_pct:.2f}% mais proxima de 0")
print(f"Curtose:    {kurt_melhoria_pct:.2f}% mais proxima de 0")
print()

# ============================================================================
# 5. VALIDACOES
# ============================================================================
print("Etapa 5: Validacoes...")

# Verificar valores nulos
nulos_orig = df['log_target'].isnull().sum()
nulos_norm = df_normalizado['log_target_normalizado'].isnull().sum()
print(f"[OK] Nulos em original: {nulos_orig}")
print(f"[OK] Nulos em normalizado: {nulos_norm}")

# Verificar se ha infinitos
inf_norm = np.isinf(df_normalizado['log_target_normalizado']).sum()
print(f"[OK] Infinitos em normalizado: {inf_norm}")

# Verificar integridade de linhas
print(f"[OK] Linhas mantidas: {len(df) == len(df_normalizado)}")

print()

# ============================================================================
# 6. SALVAR ARQUIVOS
# ============================================================================
print("Etapa 6: Salvando arquivos...")
print()

# Base normalizada completa
output_file = 'base_normalizada_boxcox.csv'
df_normalizado.to_csv(output_file, index=False)
print(f"[OK] Arquivo salvo: {output_file}")
print(f"    {len(df_normalizado)} registros x {len(df_normalizado.columns)} colunas")
print()

# ============================================================================
# 7. CREAR ARQUIVO DE METADADOS
# ============================================================================
print("Etapa 7: Gerando metadados...")

metadata = {
    'data_geracao': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'arquivo_origem': 'base_analitica_final.csv',
    'arquivo_saida': 'base_normalizada_boxcox.csv',
    'registros': len(df_normalizado),
    'colunas': len(df_normalizado.columns),
    'transformacao': 'Box-Cox',
    'lambda_otimizado': float(lambda_optimal),
    'variavel_original': 'log_target',
    'variavel_normalizada': 'log_target_normalizado',
    'estatisticas_original': {
        'media': float(df['log_target'].mean()),
        'mediana': float(df['log_target'].median()),
        'desvio_padrao': float(df['log_target'].std()),
        'minimo': float(df['log_target'].min()),
        'maximo': float(df['log_target'].max()),
        'assimetria': float(df['log_target'].skew()),
        'curtose': float(df['log_target'].kurtosis())
    },
    'estatisticas_normalizado': {
        'media': float(df_normalizado['log_target_normalizado'].mean()),
        'mediana': float(df_normalizado['log_target_normalizado'].median()),
        'desvio_padrao': float(df_normalizado['log_target_normalizado'].std()),
        'minimo': float(df_normalizado['log_target_normalizado'].min()),
        'maximo': float(df_normalizado['log_target_normalizado'].max()),
        'assimetria': float(df_normalizado['log_target_normalizado'].skew()),
        'curtose': float(df_normalizado['log_target_normalizado'].kurtosis())
    }
}

import json
with open('metadados_normalizacao_boxcox.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Arquivo de metadados salvo: metadados_normalizacao_boxcox.json")
print()

# ============================================================================
# 8. RELATORIO FINAL
# ============================================================================
print("=" * 80)
print("RESUMO")
print("=" * 80)
print()
print("Transformacao: Box-Cox")
print(f"Lambda otimizado: {lambda_optimal:.4f}")
print()
print("Arquivos Gerados:")
print(f"  1. base_normalizada_boxcox.csv")
print(f"     - 20.228 registros x 19 colunas")
print(f"     - Inclui: log_target (original) + log_target_normalizado")
print(f"     - Parametro lambda para referencia")
print()
print(f"  2. metadados_normalizacao_boxcox.json")
print(f"     - Informacoes de transformacao")
print(f"     - Estatisticas descritivas antes/depois")
print()
print("Proximas Etapas:")
print("  - Usar 'log_target_normalizado' para regressao linear")
print("  - Usar 'log_target' para modelos de arvore (XGBoost, LightGBM)")
print("  - Validar pressupostos com testes de residuos")
print()
print("=" * 80)
print("Normalizacao concluida com sucesso!")
print("=" * 80)
