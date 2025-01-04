import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from constants_config import COLOR_PALETTE_FOR_PLSR, TARGET_VARIABLES

dpi = 500
def get_best_n_components(rmse,target_variable_name):
    if not target_variable_name:
        return sorted(rmse['Avg_RMSE'], key = lambda x:x[1])[0][0]['n_components']
    return sorted(rmse[target_variable_name], key = lambda x:x[1])[0][0]['n_components']

tuning_json_files = ['multi_rmses.json', 'n_value_rmses.json', 'sc_value_rmses.json', 'st_value_rmses.json']
fig_outputs = './outputs/figs'
tuning_folder = './outputs/tuning'
tuning_results = {}

for tuning_file in tuning_json_files:
    with open(os.path.join(tuning_folder,tuning_file)) as f:
        rmses = json.load(f)
    
    target_variable_name = ''
    model_name = tuning_file.split('.')[0].replace('_rmses','')
    if 'multi' not in model_name:
        model_name = model_name.split('_')
        model_name[0] = model_name[0].upper()
        model_name[1] = model_name[1].capitalize()
        model_name = '_'.join(model_name)
        target_variable_name = model_name
    best_n_components = get_best_n_components(rmses, target_variable_name)
    model_name = model_name + '_PLSR'
    tuning_results[model_name] = best_n_components
    
    print(f'Best {model_name} Number of Components:', best_n_components)



with open(os.path.join(tuning_folder,'multi_rmses.json')) as f:
    multi_rmses = json.load(f)

multi_model_name = 'multi_PLSR'
best_n_components = tuning_results[multi_model_name]

multi_n_value_rmse = multi_rmses['N_Value']
multi_sc_value_rmse = multi_rmses['SC_Value']
multi_st_value_rmse = multi_rmses['ST_Value']
multi_avg_value_rmse = [item[1] for item in multi_rmses['Avg_RMSE']]


# Plot the RMSES as a function of the number of components
n_components = [item[0]['n_components'] for item in multi_rmses['Avg_RMSE']]

from matplotlib import cm

# Create a color palette
colors = cm.viridis([0.2, 0.4, 0.6, 0.8])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines with improved color palette
ax.plot(n_components, multi_avg_value_rmse, label='Average RMSE', color=COLOR_PALETTE_FOR_PLSR['Avg_RMSE'], linewidth=2)
ax.plot(n_components, multi_n_value_rmse, label='N Value RMSE', color=COLOR_PALETTE_FOR_PLSR['N_Value'], linewidth=2)
ax.plot(n_components, multi_sc_value_rmse, label='SC Value RMSE', color=COLOR_PALETTE_FOR_PLSR['SC_Value'], linewidth=2)
ax.plot(n_components, multi_st_value_rmse, label='ST Value RMSE', color=COLOR_PALETTE_FOR_PLSR['ST_Value'], linewidth=2)

# Add a red dot for the best number of components
ax.plot(best_n_components, multi_avg_value_rmse[n_components.index(best_n_components)], 
        'ro', label=f'Best Number of Components {best_n_components}', markersize=8)

# Set labels, title, and legend
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.legend(fontsize=10)

# Add grid and fine-tune its transparency
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the plot
fig.savefig(os.path.join(fig_outputs, 'multi_rmses_plot.png'), dpi=dpi, bbox_inches='tight')


# RMSE as a function of CV10 Folds

cv10_json_file = 'multi_rmse_cv10.json'
cv10_outputs = './outputs/cv10'

with open(os.path.join(cv10_outputs,cv10_json_file)) as f:
    multi_rmse_cv10 = json.load(f)


# Plot the RMSES as a function of the number of folds
n_folds = [i for i in range(1,11)]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines with improved color palette
ax.plot(n_folds, multi_rmse_cv10['Avg_RMSE'], label='Average RMSE', color=COLOR_PALETTE_FOR_PLSR['Avg_RMSE'], linewidth=2)
ax.plot(n_folds, multi_rmse_cv10['N_Value'], label='N Value RMSE', color=COLOR_PALETTE_FOR_PLSR['N_Value'], linewidth=2)
ax.plot(n_folds, multi_rmse_cv10['SC_Value'], label='SC Value RMSE', color=COLOR_PALETTE_FOR_PLSR['SC_Value'], linewidth=2)
ax.plot(n_folds, multi_rmse_cv10['ST_Value'], label='ST Value RMSE', color=COLOR_PALETTE_FOR_PLSR['ST_Value'], linewidth=2)
# Set labels, title, and legend
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.legend(fontsize=10)

# Add grid and fine-tune its transparency
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the plot
fig.savefig(os.path.join(fig_outputs, 'multi_rmses_cv10.png'), dpi=dpi, bbox_inches='tight')



# PLOT RMSE on test set

test_json_file = 'multi_rmse_test.json'
test_outputs = './outputs/test'
res = {}
with open(os.path.join(test_outputs,test_json_file)) as f:
    multi_rmse_test = json.load(f)
    for idx,val in enumerate(multi_rmse_test):
        res[TARGET_VARIABLES[idx]] = val

    res['Avg_RMSE'] = np.mean(multi_rmse_test)

# Create a bar plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot each bar with the corresponding color from the palette
bars = ax.bar(res.keys(), res.values(), color=[COLOR_PALETTE_FOR_PLSR[key] for key in res.keys()])

# Add labels for the legend
labels = ['N_Value', 'SC_Value', 'ST_Value', 'Avg_RMSE']
for label in labels:
    bars[labels.index(label)].set_label(label)


ax.set_xlabel('Target Variables', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.legend(fontsize=10)
fig.savefig(os.path.join(fig_outputs, 'multi_rmses_test.png'), dpi=dpi, bbox_inches='tight')



        



