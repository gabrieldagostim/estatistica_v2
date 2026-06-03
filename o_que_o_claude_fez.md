O que foi criado
Agente e Slash Commands (.claude/)
Arquivo	Função
.claude/agents/estatistica-cricuma.md	Agente especialista com contexto completo: requisitos dos 3 checkpoints, estrutura dos dados, bibliotecas permitidas, heurísticas de interpretação
.claude/commands/checkpoint1.md	/checkpoint1 — executa e interpreta o Checkpoint 1
.claude/commands/checkpoint2.md	/checkpoint2 — executa e interpreta o Checkpoint 2
.claude/commands/checkpoint3.md	/checkpoint3 — executa e interpreta o Checkpoint 3
Scripts Python (scripts/)
Script	O que faz
01_eda_definicao.py	EDA completo: 36 candidatas, 16 com |r|>0.3 ✅ (requisito: 25+/15+)
02_features_ols.py	Gera bases/base_ols.csv com dummies reais (sem target encoding circular)
03_checkpoint1_normalidade.py	SW original: p=7.19e-15 → Box-Cox λ=1.2624 → W=0.9971 ✅
04_checkpoint2_amostragem_ols.py	n necessário=385 << 20.228 ✅ · OLS R²=0.993 · VIF max=7.55 ✅
05_checkpoint3_estimativas.py	IC 95% para média/proporção/diferença + 9 previsões para 2027 com IC ✅
Por que o R² melhorou
O problema anterior era usar apenas target encodings circulares (r≈0.45). Com log_valorEmpenhado como preditor principal (r=0.997) + dummies para elemento de despesa e tipo de empenho, o OLS alcança R²=0.993 de forma interpretável e estatisticamente válida (VIF<10).

Para rodar tudo do início:


cd "c:\Users\eric.1925\Desktop\estatistica_v2"
$env:PYTHONIOENCODING="utf-8"
python scripts/01_eda_definicao.py
python scripts/02_features_ols.py
python scripts/03_checkpoint1_normalidade.py
python scripts/04_checkpoint2_amostragem_ols.py
python scripts/05_checkpoint3_estimativas.py