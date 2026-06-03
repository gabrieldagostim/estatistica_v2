import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor

# ========================================================= #
# 1. CARREGAR BASE                                          #
# ========================================================= # 

df = pd.read_csv('base_normalizada_boxcox.csv')

# =========================================================
# 2. DEFINIR TARGET
# =========================================================

# Para modelos de árvore:
TARGET = 'log_target'

# Para regressão linear:
# TARGET = 'log_target_normalizado'

# =========================================================
# 3. REMOVER COLUNAS DESNECESSÁRIAS
# =========================================================

colunas_remover = [
    'valorPagoEmpenho',
    'log_target',
    'log_target_normalizado',
    'lambda_boxcox'
]

features = [c for c in df.columns if c not in colunas_remover]

X = df[features]
y = df[TARGET]

# =========================================================
# 4. TREINO / TESTE
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# 5. MODELO
# =========================================================

modelo = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# =========================================================
# 6. TREINAR
# =========================================================

modelo.fit(X_train, y_train)

# =========================================================
# 7. PREDIÇÕES
# =========================================================

pred_log = modelo.predict(X_test)

# =========================================================
# 8. CONVERTER PARA ESCALA REAL
# =========================================================

# Caso use log_target:
pred_real = np.exp(pred_log)
y_real = np.exp(y_test)

# =========================================================
# 9. MÉTRICAS
# =========================================================

mae = mean_absolute_error(y_real, pred_real)

rmse = np.sqrt(mean_squared_error(y_real, pred_real))

r2 = r2_score(y_real, pred_real)

print('\n========== RESULTADOS ==========')
print(f'MAE  : {mae:,.2f}')
print(f'RMSE : {rmse:,.2f}')
print(f'R²   : {r2:.4f}')

# =========================================================
# 10. IMPORTÂNCIA DAS FEATURES
# =========================================================

importancias = pd.DataFrame({
    'feature': X.columns,
    'importance': modelo.feature_importances_
})

importancias = importancias.sort_values(
    by='importance',
    ascending=False
)

print('\n========== TOP FEATURES ==========')
print(importancias.head(15))

# =========================================================
# 11. SALVAR MODELO
# =========================================================

import joblib

joblib.dump(modelo, 'modelo_xgboost.pkl')

print('\nModelo salvo com sucesso!')