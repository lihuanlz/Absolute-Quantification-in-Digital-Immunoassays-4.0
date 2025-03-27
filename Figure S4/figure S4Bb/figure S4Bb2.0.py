


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

N = 2000000
Ms_values = np.logspace(np.log10(1), np.log10(N*np.exp(1/N)), num=100)
M_N_ratios = np.logspace(np.log10(1/N), np.log10(-np.log(1/N)), num=100)

results = {}
min_values = {}


for Ms in Ms_values:
    results[Ms] = []
    min_val = float('inf')
    min_ratio = None
    for M_N_ratio in M_N_ratios:
        C1 = (1 - np.exp(-M_N_ratio)) / (1 - np.exp(-Ms/N))
        C2 = M_N_ratio * N / Ms
        expression_value = np.abs((C1 - C2) / C2)
        results[Ms].append(expression_value)
        if expression_value < min_val:
            min_val = expression_value
            min_ratio = M_N_ratio
    min_values[Ms] = (min_val, min_ratio)


Ms_array = np.array(Ms_values)
M_N_ratios_array = np.array(M_N_ratios)
expression_values_array = np.array([results[Ms] for Ms in Ms_values])


plt.figure(figsize=(10, 8))

X, Y = np.meshgrid(M_N_ratios_array, Ms_array)
Z = expression_values_array


scatter = plt.scatter(X.flatten(), Y.flatten(), c=Z.flatten(), cmap='tab10', s=10, alpha=1)


cbar = plt.colorbar(scatter, label='Relative Error')
cbar.ax.tick_params(labelsize=15)  
cbar.set_label('Relative Error', fontsize=15)  


cbar.set_ticks(np.linspace(Z.min(), Z.max(), 11))
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

contour_lines = plt.contour(X, Y, Z, levels=[0.05, 0.1], colors=['blue', 'red'], linewidths=2)


plt.xscale('log')
plt.yscale('log')


plt.xlabel('M/N ratio', fontsize=15)
plt.ylabel('Ms', fontsize=15)
plt.title(f'Scatter Plot with Relative Error\nN={N}', fontsize=15)


plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()

plt.savefig('figure_S7A_scatter_with_contours.svg', dpi=300)
plt.show()


csv_data = []
for Ms, (min_val, min_ratio) in min_values.items():
    csv_data.append([Ms, min_ratio, min_val])
df = pd.DataFrame(csv_data, columns=['Ms', 'min_ratio', 'expression_value'])
df.to_csv('5_expression_values2.csv', index=False)
