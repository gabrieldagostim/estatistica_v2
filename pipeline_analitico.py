"""
Pipeline analitico - Transparencia Municipio de Criciuma
Base: Despesas com Pessoal (2012-2026) -> 20.691 registros
Enriquecimento: Programas/Acoes, Cargos/Vencimentos, Adiantamentos
"""
import json, glob, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

DADOS = 'dados'

# ----------------------------------------------------------------------
# 1. CARGA DOS ARQUIVOS
# ----------------------------------------------------------------------
def carrega(prefixo):
    frames = []
    for f in sorted(glob.glob(os.path.join(DADOS, f'{prefixo}*.json'))):
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        frames.append(pd.json_normalize(data, max_level=1))
    return pd.concat(frames, ignore_index=True)

print('Carregando arquivos...')
dp  = carrega('Despesas com Pessoal')
dpa = carrega('Despesas por Programas')
cv  = carrega('Cargos e Vencimentos')
ad  = carrega('Adiantamentos')
print(f'  Despesas Pessoal      : {len(dp):>6} regs, {dp.shape[1]} cols')
print(f'  Programas/Acoes       : {len(dpa):>6} regs, {dpa.shape[1]} cols')
print(f'  Cargos/Vencimentos    : {len(cv):>6} regs, {cv.shape[1]} cols')
print(f'  Adiantamentos         : {len(ad):>6} regs, {ad.shape[1]} cols')

# ----------------------------------------------------------------------
# 2. LIMPEZA / TIPOS DA BASE PRINCIPAL
# ----------------------------------------------------------------------
dp['dataEmpenho'] = pd.to_datetime(dp['dataEmpenho'], errors='coerce')
dp['anoExercicio'] = pd.to_numeric(dp['anoExercicio'], errors='coerce')
num_cols = ['valorEmpenho','valorEmpenhado','valorPagoEmpenho','valorLiquidadoEmpenho',
            'saldoAPagar','saldoALiquidar','saldoAPagarLiquidado',
            'valorRestosAPagarProcessados','valorRestosAPagarNaoProcessados',
            'valorRestosAPagarCancelados','ValorAnuladoEmpenho','valorPagoRestosEmpenho']
for c in num_cols:
    dp[c] = pd.to_numeric(dp[c], errors='coerce').fillna(0)

# Filtra registros validos para a variavel-alvo
dp = dp[dp['valorPagoEmpenho'] > 0].copy()
print(f'\nApos filtro valorPagoEmpenho > 0: {len(dp)} registros')

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING - TEMPORAIS / ESTRUTURAIS
# ----------------------------------------------------------------------
dp['mes_empenho']      = dp['dataEmpenho'].dt.month
dp['trimestre']        = dp['dataEmpenho'].dt.quarter
dp['dia_semana']       = dp['dataEmpenho'].dt.dayofweek
dp['dia_mes']          = dp['dataEmpenho'].dt.day
dp['fim_ano']          = (dp['mes_empenho'] >= 11).astype(int)
dp['inicio_ano']       = (dp['mes_empenho'] <= 2).astype(int)

# contagens de listas aninhadas
dp['n_pagamentos']     = dp['pagamentos'].apply(lambda x: len(x) if isinstance(x,list) else 0)
dp['n_liquidacoes']    = dp['liquidacoes'].apply(lambda x: len(x) if isinstance(x,list) else 0)
dp['n_docfiscais']     = dp['documentosFiscais'].apply(lambda x: len(x) if isinstance(x,list) else 0)

# credor / dotacao ja vem achatado por json_normalize(level=1)
# colunas como 'credor.cnpjCpfCredor' existem; nao sao usadas diretamente nas features

# razoes / proxies
dp['tx_anulacao']      = dp['ValorAnuladoEmpenho'] / (dp['valorEmpenhado'].abs()+1)
dp['tx_restos']        = (dp['valorRestosAPagarProcessados']+dp['valorRestosAPagarNaoProcessados']) / (dp['valorEmpenhado'].abs()+1)
dp['log_saldoAPagar']  = np.log1p(dp['saldoAPagar'].clip(lower=0))

# ----------------------------------------------------------------------
# 4. AGREGACOES DE TABELAS COMPLEMENTARES
# ----------------------------------------------------------------------
# 4.1 Programas/Acoes -> chave (ano, descricaoPrograma, descricaoOrgao)
dpa['ano'] = pd.to_numeric(dpa['ano'], errors='coerce')
for c in ['valorOrcado','valorOrcadoAtualizado','valorEmpenhado','valorLiquidado','valorPago',
         'percentualEmpenhadoSobreOrcadoAtualizado','percentualPagoSobreOrcadoAtualizado']:
    dpa[c] = pd.to_numeric(dpa[c], errors='coerce').fillna(0)

prog_agg = dpa.groupby(['ano','descricaoPrograma','descricaoOrgao']).agg(
    prog_valor_orcado=('valorOrcadoAtualizado','sum'),
    prog_valor_empenhado=('valorEmpenhado','sum'),
    prog_valor_pago=('valorPago','sum'),
    prog_pct_empenhado=('percentualEmpenhadoSobreOrcadoAtualizado','mean'),
    prog_pct_pago=('percentualPagoSobreOrcadoAtualizado','mean'),
    prog_n_acoes=('idAcao','nunique'),
).reset_index().rename(columns={'ano':'anoExercicio'})

# 4.2 Cargos/Vencimentos -> aggregate (ano, nomeEntidade)
cv['ano'] = pd.to_numeric(cv['ano'], errors='coerce')
def soma_vagas(qv, key):
    if not isinstance(qv,list): return 0
    return sum((q.get(key,0) or 0) for q in qv)
cv['vagas_criadas']    = cv['quadroVagas'].apply(lambda x: soma_vagas(x,'quantidadeVagasCriadas'))
cv['vagas_preenchidas']= cv['quadroVagas'].apply(lambda x: soma_vagas(x,'quantidadeVagasPreenchidas'))
def media_salarial(ns):
    if not isinstance(ns,list) or len(ns)==0: return np.nan
    vals = [n.get('valor',0) or 0 for n in ns if isinstance(n,dict)]
    return np.mean(vals) if vals else np.nan
cv['salario_medio_cargo'] = cv['niveisSalariais'].apply(media_salarial)

cv_agg = cv.groupby(['ano','nomeEntidade']).agg(
    cv_n_cargos=('cargo','nunique'),
    cv_vagas_criadas=('vagas_criadas','sum'),
    cv_vagas_preenchidas=('vagas_preenchidas','sum'),
    cv_salario_medio=('salario_medio_cargo','mean'),
).reset_index().rename(columns={'ano':'anoExercicio'})
cv_agg['cv_taxa_ocupacao'] = cv_agg['cv_vagas_preenchidas']/(cv_agg['cv_vagas_criadas']+1)

# 4.3 Adiantamentos -> (ano, orgao)
ad['dataEmpenho'] = pd.to_datetime(ad['dataEmpenho'], errors='coerce')
ad['ano'] = ad['dataEmpenho'].dt.year
ad['valorEmpenho'] = pd.to_numeric(ad['valorEmpenho'], errors='coerce').fillna(0)
ad_agg = ad.groupby(['ano','orgao']).agg(
    ad_n_adiantamentos=('numero','count'),
    ad_valor_total=('valorEmpenho','sum'),
).reset_index().rename(columns={'ano':'anoExercicio','orgao':'descricaoOrgao'})

# ----------------------------------------------------------------------
# 5. JOINS
# ----------------------------------------------------------------------
dp = dp.merge(prog_agg, on=['anoExercicio','descricaoPrograma','descricaoOrgao'], how='left')
dp = dp.merge(cv_agg,   on=['anoExercicio','nomeEntidade'], how='left')
dp = dp.merge(ad_agg,   on=['anoExercicio','descricaoOrgao'], how='left')

for c in ['prog_valor_orcado','prog_valor_empenhado','prog_valor_pago','prog_pct_empenhado',
          'prog_pct_pago','prog_n_acoes','cv_n_cargos','cv_vagas_criadas','cv_vagas_preenchidas',
          'cv_salario_medio','cv_taxa_ocupacao','ad_n_adiantamentos','ad_valor_total']:
    dp[c] = dp[c].fillna(0)

print(f'\nApos joins: {len(dp)} registros, {dp.shape[1]} colunas')

# ----------------------------------------------------------------------
# 6. TARGET ENCODING (k-fold simples) p/ categoricas de alta cardinalidade
# ----------------------------------------------------------------------
TARGET = 'valorPagoEmpenho'
dp['log_target'] = np.log1p(dp[TARGET])

def stat_encode(df, cols, target, prefix):
    """Substitui cada linha pela media/mediana/std/max do grupo definido por cols."""
    g = df.groupby(cols)[target].agg(['mean','median','std','max','count'])
    g.columns = [f'{prefix}_{s}' for s in g.columns]
    return df.merge(g.reset_index(), on=cols, how='left')

# encodings simples
for c in ['descricaoOrgao','descricaoPrograma','descricaoFuncao','descricaoSubfuncao',
          'descricaoElemento','tipoRecurso','descricaoUnidade','tipoEmpenho',
          'descricaoDetalhamentoElemento','modalidadeAplicacao']:
    dp = stat_encode(dp, [c], 'log_target', f'te_{c}')

# encodings de interacao (mais granulares)
dp = stat_encode(dp, ['descricaoOrgao','descricaoElemento'],   'log_target', 'te_orgEle')
dp = stat_encode(dp, ['descricaoPrograma','descricaoElemento'],'log_target', 'te_progEle')
dp = stat_encode(dp, ['descricaoUnidade','descricaoElemento'], 'log_target', 'te_uniEle')
dp = stat_encode(dp, ['descricaoOrgao','descricaoFuncao'],     'log_target', 'te_orgFun')
dp = stat_encode(dp, ['descricaoOrgao','tipoRecurso'],         'log_target', 'te_orgRec')
dp = stat_encode(dp, ['descricaoFuncao','descricaoElemento'],  'log_target', 'te_funEle')
dp = stat_encode(dp, ['descricaoSubfuncao','descricaoElemento'],'log_target','te_subEle')
dp = stat_encode(dp, ['descricaoUnidade','descricaoFuncao'],   'log_target', 'te_uniFun')
dp = stat_encode(dp, ['descricaoOrgao','descricaoPrograma','descricaoElemento'],'log_target','te_orgProgEle')

# credor / despesa-dotacao (alta cardinalidade -> sinal forte)
dp['cnpjCpfCredor']    = dp['credor.cnpjCpfCredor'] if 'credor.cnpjCpfCredor' in dp.columns else None
dp['idDespesaDot']     = dp['dotacaoOrcamentaria.idDespesa'] if 'dotacaoOrcamentaria.idDespesa' in dp.columns else None
if dp['cnpjCpfCredor'] is not None:
    dp = stat_encode(dp, ['cnpjCpfCredor'], 'log_target', 'te_credor')
if dp['idDespesaDot'] is not None:
    dp = stat_encode(dp, ['idDespesaDot'], 'log_target', 'te_despesaDot')

# ----------------------------------------------------------------------
# 7. LAGS / ROLLING -- por orgao, por ano
# ----------------------------------------------------------------------
ano_org = dp.groupby(['anoExercicio','descricaoOrgao'])['valorPagoEmpenho'].agg(['sum','mean','count']).reset_index()
ano_org = ano_org.sort_values(['descricaoOrgao','anoExercicio'])
ano_org['orgao_lag1_sum']  = ano_org.groupby('descricaoOrgao')['sum'].shift(1)
ano_org['orgao_roll3_sum'] = ano_org.groupby('descricaoOrgao')['sum'].shift(1).rolling(3,min_periods=1).mean().reset_index(0,drop=True)
ano_org['orgao_lag1_mean'] = ano_org.groupby('descricaoOrgao')['mean'].shift(1)
ano_org = ano_org[['anoExercicio','descricaoOrgao','orgao_lag1_sum','orgao_roll3_sum','orgao_lag1_mean']]
dp = dp.merge(ano_org, on=['anoExercicio','descricaoOrgao'], how='left')

ano_prog = dp.groupby(['anoExercicio','descricaoPrograma'])['valorPagoEmpenho'].agg(['sum','mean']).reset_index()
ano_prog = ano_prog.sort_values(['descricaoPrograma','anoExercicio'])
ano_prog['prog_lag1_sum']  = ano_prog.groupby('descricaoPrograma')['sum'].shift(1)
ano_prog['prog_roll3_sum'] = ano_prog.groupby('descricaoPrograma')['sum'].shift(1).rolling(3,min_periods=1).mean().reset_index(0,drop=True)
ano_prog = ano_prog[['anoExercicio','descricaoPrograma','prog_lag1_sum','prog_roll3_sum']]
dp = dp.merge(ano_prog, on=['anoExercicio','descricaoPrograma'], how='left')

# rankings
dp['rank_pago_orgao_ano'] = dp.groupby(['anoExercicio','descricaoOrgao'])['valorPagoEmpenho'].rank(pct=True)

# ----------------------------------------------------------------------
# 8. SELECAO DE FEATURES E CORRELACAO
# ----------------------------------------------------------------------
# colunas geradas pelo stat_encode  (auto-deteccao)
te_cols = [c for c in dp.columns if c.startswith('te_')]
candidatas = sorted(set([
    # estruturais agregadas
    'prog_valor_orcado','prog_valor_empenhado','prog_valor_pago',
    'prog_pct_empenhado','prog_pct_pago','prog_n_acoes',
    'cv_n_cargos','cv_vagas_criadas','cv_vagas_preenchidas','cv_salario_medio','cv_taxa_ocupacao',
    'ad_n_adiantamentos','ad_valor_total',
    # temporais
    'anoExercicio','mes_empenho','trimestre','dia_mes','fim_ano','inicio_ano',
    # contagens
    'n_pagamentos','n_liquidacoes','n_docfiscais',
    # razoes
    'tx_anulacao','tx_restos','log_saldoAPagar',
    # lags / rolling
    'orgao_lag1_sum','orgao_roll3_sum','orgao_lag1_mean',
    'prog_lag1_sum','prog_roll3_sum',
] + te_cols))
print(f'\nTotal candidatas: {len(candidatas)}')

corr = dp[candidatas + ['log_target']].apply(pd.to_numeric, errors='coerce').corr()['log_target'].drop('log_target')
corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)

print('\n========= CORRELACAO COM log(valorPagoEmpenho) =========')
for v, c in corr_sorted.items():
    flag = '***' if abs(c) > 0.3 else '   '
    print(f'  {flag} {v:35s}: {c:+.4f}')

selecionadas = corr_sorted[corr_sorted.abs() > 0.3]
print(f'\n>>> Variaveis com |corr|>0.3 : {len(selecionadas)}')

# ----------------------------------------------------------------------
# 9. MULTICOLINEARIDADE
# ----------------------------------------------------------------------
print('\n========= MULTICOLINEARIDADE (|corr|>0.85 entre selecionadas) =========')
sel = list(selecionadas.index)
M = dp[sel].apply(pd.to_numeric, errors='coerce').corr().abs()
pares = []
for i in range(len(sel)):
    for j in range(i+1, len(sel)):
        if M.iloc[i,j] > 0.85:
            pares.append((sel[i], sel[j], M.iloc[i,j]))
for a,b,v in sorted(pares, key=lambda x:-x[2]):
    print(f'  {a:35s} <-> {b:35s} = {v:.3f}')

# ----------------------------------------------------------------------
# 10. PERSISTENCIA
# ----------------------------------------------------------------------
out_cols = ['valorPagoEmpenho','log_target'] + sel
dp[out_cols].to_csv('base_analitica_final.csv', index=False)
print(f'\nBase analitica salva: base_analitica_final.csv ({len(dp)} regs, {len(out_cols)} cols)')

# tabela resumo correlacoes
pd.DataFrame({'variavel': corr_sorted.index, 'correlacao': corr_sorted.values,
              'selecionada': corr_sorted.abs() > 0.3}).to_csv('correlacoes.csv', index=False)
print('Tabela correlacoes salva: correlacoes.csv')
