


import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import griddata
import math
from matplotlib.ticker import MaxNLocator  
import matplotlib.ticker as ticker
def calculate_max_balls_on_sphere(big_radius, small_radius):
    max_balls = (4 * math.pi * big_radius ** 2) / (3 * small_radius ** 2)
    return max_balls

magnetic_beads = 2000000
N = magnetic_beads
Y = 100000
error_limit = 0.2


n_values = np.linspace(1, N, num=100)
postive_ratios = np.logspace(np.log10(1/N), np.log10(1-1/N), num=100)

results = []





       










        







for postive_ratio in postive_ratios:

    for n in n_values:
        M_mean = -math.log(1 - postive_ratio)*N
        
        M_var = ((N**2) / ((1-postive_ratio)**2)) * ((1-n/N) * postive_ratio * (1-postive_ratio) / n + postive_ratio * (1-postive_ratio) / N) + \
                0.5 * ((N**2) / ((1-postive_ratio)**4)) * ((1-n/N) * postive_ratio * (1-postive_ratio) / n + postive_ratio * (1-postive_ratio) / N)**2
        M_std = np.sqrt(M_var)
        
        Y_mean = -math.log(1 - Y / N)*N
        
        Y_var = ((N**2) / ((1-Y/N)**2)) * ((1-n/N) * (Y/N) * (1-Y/N) / n + (Y/N) * (1-Y/N) / N) + \
                0.5 * ((N**2) / ((1-Y/N)**4)) * ((1-n/N) * (Y/N) * (1-Y/N) / n + (Y/N) * (1-Y/N) / N)**2
        Y_std = np.sqrt(Y_var)
        
        if Y_mean == 0 or M_mean == 0:
            relative_error = np.nan
        else:
            relative_error = np.sqrt((M_std / M_mean) ** 2 + (Y_std / Y_mean) ** 2)
        
        if relative_error < error_limit:
            results.append((postive_ratio, n / N, relative_error))










postive_ratios, n_N_values, relative_errors = zip(*results)

lambda_values = -np.log(1 - np.array(postive_ratios))


csv_filename = '6_5sampling_results_log_range_M_N_filtered2.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['postive_ratio', 'n / N', 'lambda_values', 'relative_error'])
    for result, lambda_value in zip(results, lambda_values):
        writer.writerow([result[0], result[1], lambda_value, result[2]])

print(f'Results written to {csv_filename}')


lambda_grid, n_N_grid = np.meshgrid(np.logspace(np.log10(min(lambda_values)), np.log10(max(lambda_values)), 100),
                                    np.linspace(min(n_N_values), max(n_N_values), 100))


relative_error_grid = griddata((lambda_values, n_N_values), relative_errors, (lambda_grid, n_N_grid), method='cubic')


plt.figure(figsize=(10, 8))


scatter = plt.scatter(lambda_values, n_N_values, c=relative_errors, cmap='tab10', s=20, edgecolors='w', alpha=1, marker='o')


cbar = plt.colorbar(scatter)  
cbar.set_label('Sampling Error', fontsize=15)  


cbar.locator = MaxNLocator(nbins=10)  
cbar.update_ticks()  
cbar.set_ticks(np.linspace(np.min(relative_errors), np.max(relative_errors), 11))
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

cbar.ax.tick_params(labelsize=15)  


plt.title(f'Relative Error Scatter Plot (Sampling Error)\nN={N}, Ms={Y}', fontsize=15)
plt.xlabel('λ', fontsize=15)
plt.ylabel('n/N', fontsize=15)
plt.xscale('log')


CS = plt.contour(lambda_grid, n_N_grid, relative_error_grid, levels=[0.05, 0.1], colors='black', linestyles='-')
plt.clabel(CS, inline=True, fontsize=15)


plt.axvline(x=131 / N, color='black', linestyle='--', label='131/N')
plt.axvline(x=1.204, color='red', linestyle='--', label='1.204')


plt.legend()


plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


dpi_value = 300
plt.savefig('scatter_plot_with_contours.Svg', dpi=dpi_value)


plt.show()
