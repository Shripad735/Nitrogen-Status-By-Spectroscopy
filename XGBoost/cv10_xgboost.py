import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from constants_config import COLOR_PALETTE_FOR_PLSR
import numpy as np
import os

fig_outputs = './figures'
xgb_model = joblib.load('./models/xgboost_multi_output_plsr/model.pkl')
print(xgb_model)
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
target_vars = ['N_Value','SC_Value','ST_Value']
model = xgb_model.model
train_dataset = xgb_model.train_data
X = train_dataset.drop(columns = ['ID','N_Value','SC_Value','ST_Value'])
Y = train_dataset[target_vars]

rmses = {target:[] for target in target_vars}
rmses['Avg_RMSE'] = []
for train_index, val_index in tqdm(kf.split(X), total=n_splits, desc='Cross Validation'):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = Y.iloc[train_index], Y.iloc[val_index]


    model.fit(X_train_fold, y_train_fold)
    y_hat = model.predict(X_val_fold)
    n_value_rmse,sc_value_rmse,st_value_rmse = np.sqrt(mean_squared_error(y_true=y_val_fold,y_pred=y_hat, multioutput='raw_values'))
    rmses['N_Value'] += [n_value_rmse]
    rmses['SC_Value'] += [sc_value_rmse]
    rmses['ST_Value'] += [st_value_rmse]
    rmses['Avg_RMSE'] += [np.mean([n_value_rmse,sc_value_rmse,st_value_rmse])]

dpi = 500
# Plot the RMSES as a function of the number of folds
n_folds = [i for i in range(1,11)]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines with improved color palette
ax.plot(n_folds, rmses['Avg_RMSE'], label='Average RMSE', color=COLOR_PALETTE_FOR_PLSR['Avg_RMSE'], linewidth=2)
ax.plot(n_folds, rmses['N_Value'], label='N Value RMSE', color=COLOR_PALETTE_FOR_PLSR['N_Value'], linewidth=2)
ax.plot(n_folds, rmses['SC_Value'], label='SC Value RMSE', color=COLOR_PALETTE_FOR_PLSR['SC_Value'], linewidth=2)
ax.plot(n_folds, rmses['ST_Value'], label='ST Value RMSE', color=COLOR_PALETTE_FOR_PLSR['ST_Value'], linewidth=2)
# Set labels, title, and legend
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.legend(fontsize=10)

# Add grid and fine-tune its transparency
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the plot
fig.savefig(os.path.join(fig_outputs, 'multi_xgb_plsr_cv10.png'), dpi=dpi, bbox_inches='tight')


