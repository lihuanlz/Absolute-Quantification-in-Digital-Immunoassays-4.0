

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv
from scipy.special import i0


N = 2000000
Ms = 100000


M_N_ratios = np.logspace(np.log10(1/N), np.log10(-np.log(1/N)), num=N) 


max_postive_beads = 0


results = []


for M_N_ratio in M_N_ratios:  
    C1 = (1-np.exp(-M_N_ratio)) / (1-np.exp(-Ms/N))
    
    C2 = M_N_ratio * N / Ms
    expression_value = np.abs((C1 - C2) / C2)
    
    
    results.append([M_N_ratio, expression_value])

    if expression_value < 0.1:
        max_postive_beads = M_N_ratio * N
    else:
        break


with open('expression_values.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['M/N ratio', 'Expression Value'])  
    for result in results:
        writer.writerow(result)

print('CSV file has been written with the expression values.')



x_values = []
y_values = []


for M in range(1, int(-N*np.log(1/N)) + 1):
    x = M/N
    y = 1 - np.exp(-M/N)
    x_values.append(x)
    y_values.append(y)


plt.figure(figsize=(10, 9))
ax_main = plt.subplot(111)
ax_main.plot(x_values, y_values, linestyle='-', markersize=1, label='(1 - exp(-M/N)) / N vs M/N')
ax_main.set_xlabel('λ=M/N', fontsize=15)

ax_main.set_ylabel('Postive %', fontsize=15)
ax_main.set_title(f'Postive % vs λ\nN={N},Ms={Ms}', fontsize=15)
max_postive_beads=int(max_postive_beads)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


x_highlight = max_postive_beads / N
y_highlight = (1 - np.exp(-max_postive_beads / N)) 
ax_main.plot(x_highlight, y_highlight, 'ro')
ax_main.annotate(f'M={max_postive_beads}', (x_highlight, y_highlight), textcoords="offset points", xytext=(10,-10), ha='center')



x_vals_to_mark = [1.2040, -np.log(1/N)]
labels = [r'$1.2040$', r'$-\ln(1/N)$']
colors = ['black', 'purple']


offsets = [(0, 10), (20, 20), (40, 40)]  

for x_val, label, color in zip(x_vals_to_mark, labels, colors):
    ax_main.axvline(x=x_val, color=color, linestyle='--', label=f'x = {label}')
    ax_main.annotate(f'{label}', (x_val, 0.5), textcoords="offset points", xytext=(0,10), ha='center', color=color)




ax_inset = inset_axes(ax_main, width="50%", height="50%", loc='center')
x_zoomed = x_values[:max_postive_beads]
y_zoomed = y_values[:max_postive_beads]
ax_inset.plot(x_zoomed, y_zoomed, linestyle='-', color='r')
ax_inset.set_xlabel('λ')
ax_inset.set_ylabel('Postive %')
ax_inset.set_title(f'Postive % vs λ\nMax_postive_beads={max_postive_beads}')


slope, intercept, r_value, p_value, std_err = linregress(x_zoomed, y_zoomed)

y_regress = np.array(x_zoomed) * slope + intercept

ax_inset.plot(x_zoomed, y_regress, 'b--', label='Linear regression')


regress_eq = f'y = {slope:.2e}x + {intercept:.2e}\nR^2 = {r_value**2:.2f}'
ax_inset.annotate(regress_eq, xy=(0.5, 0.2), xycoords='axes fraction', ha='center', va='center', fontsize=9, bbox=dict(boxstyle="round", alpha=0.5, color="w"))


dpi_value = 300
plt.legend()

plt.savefig('figure 3A and Extended Data Figure 3.svg', dpi=dpi_value)  
plt.show()
