import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.lines as mlines


N_sum = 500000



n_values = np.linspace(100, N_sum, num=50, endpoint=True)
n_values = n_values[(n_values >= 1) & (n_values <= N_sum)]


k_values_list = []
for n in n_values:
    k_max = n - 1
    if k_max < 1:
        continue
    k_values = np.logspace(1, np.log10(k_max), num=50, endpoint=True)
    k_values = k_values[k_values >= 1]
    k_values_list.append(k_values)


results = []
for i, n in enumerate(n_values):
    for k in k_values_list[i]:
        results.append({'N': N_sum, 'n': n, 'k': k})

df_results = pd.DataFrame(results)



def calculate_M_statistics_second_order(row, alpha=0.05):
    
    try:
        N = int(row['N'])
        n = int(row['n'])
        k = int(row['k'])
    except KeyError as e:
        raise ValueError(f"Missing required column: {e}")

    
    E_M, Var_M, CV = float('nan'), float('nan'), float('nan')

    
    if n <= 0 or k < 0 or k > n or N <= 0:
        return pd.Series({'E_M': E_M, 'Var_M': Var_M, 'CV': CV})

    
    p = k / n
    if p >= 1.0:  
        E_M = float('inf')
        Var_M = float('inf')
        CV = float('nan')
        return pd.Series({'E_M': E_M, 'Var_M': Var_M, 'CV': CV})

    
    try:
        E_M = -N * math.log(1 - p)
    except ValueError:
        return pd.Series({'E_M': E_M, 'Var_M': Var_M, 'CV': CV})

    
    try:
        
        term_hyper = (N - n) / (N - 1) * (p * (1 - p)) / n  
        term_binom = (p * (1 - p)) / N  
        var_p_total = term_hyper + term_binom  
        term1 = (N**2 / (1 - p)**2) * var_p_total  
        term2 = 0.5 * (N**2 / (1 - p)**4) * var_p_total**2  
        Var_M = term1 + term2
        if Var_M < 0:  
            Var_M = float('nan')
    except ZeroDivisionError:
        return pd.Series({'E_M': E_M, 'Var_M': Var_M, 'CV': CV})

    
    try:
        CV = math.sqrt(Var_M) / E_M if E_M != 0 else float('inf')
    except ZeroDivisionError:
        CV = float('inf')

    return pd.Series({'E_M': E_M, 'Var_M': Var_M, 'CV': CV})



df_results[['E_M', 'Var_M', 'CV']] = df_results.apply(calculate_M_statistics_second_order, axis=1)

















points_x2 = [1,3,5,22,38,237,385,1787,4036,15634,24836]  
points_y2 = [62500,34884,50505,53269,53296,53127,47049,52867,53200,51012,55774]  

points_y2_normalized = [y / N_sum for y in points_y2]


def convert_x(x_list, y_list, N_sum):
    converted = []
    for x, y in zip(x_list, y_list):
        ratio = x / y
        term = -math.log(1 - ratio) * N_sum
        converted.append(round(term, 4))  
    return converted



converted_x2 = convert_x(points_x2, points_y2, N_sum)









plt.figure(figsize=(10, 8))
sc = plt.scatter(df_results['E_M'], df_results['n'], c=df_results['CV'], cmap='tab10', s=20)  
cbar = plt.colorbar(sc, label='CV')
cbar.ax.tick_params(labelsize=25)
cbar.set_label('CV', fontsize=25)
cbar.set_ticks(np.linspace(df_results['CV'].min(), df_results['CV'].max(), 11))
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


p_kn = 0.7
fixed_E_M = N_sum * (-math.log(1 - p_kn))
n_min = df_results['n'].min()
n_max = df_results['n'].max()
n_line = np.logspace(np.log10(n_min), np.log10(n_max), 100)
x_line = np.full_like(n_line, fixed_E_M)
y_line = n_line  
line_kn = plt.plot(x_line, y_line, color='black', linestyle='--', label='k/n=0.7')[0]



group2 = plt.scatter(converted_x2, points_y2, color='red', s=40, label='Droplet')  


plt.xscale('log')

plt.ylim(df_results['n'].min(), df_results['n'].max())  
plt.autoscale(enable=False)


confidence_intervals = pd.read_csv('results.csv')


ax = plt.gca()

for _, row in confidence_intervals.iterrows():
    n_number = row['n']
    final_lower = row['ci_lower']*N_sum
    final_upper = row['ci_upper']*N_sum

    lower_k = final_lower
    upper_k = final_upper
    y_value = n_number  
    x_center = (lower_k + upper_k) / 2
    x_err = (upper_k - lower_k) / 2
    ax.errorbar(x=x_center, y=y_value, xerr=x_err, yerr=None, 
                fmt='none', ecolor='black', capsize=10, zorder=1)


n_grid = np.logspace(2, np.log10(N_sum), num=100)
k_grid = np.logspace(1, np.log10(N_sum-1), num=100)
n_grid, k_grid = np.meshgrid(n_grid, k_grid)


p_grid = k_grid / n_grid
x_grid = N_sum * (-np.log(1 - p_grid))
x_grid[p_grid >= 1] = np.nan  

CV_grid = np.zeros_like(n_grid)
for i in range(n_grid.shape[0]):
    for j in range(n_grid.shape[1]):
        n = n_grid[i,j]
        k = k_grid[i,j]
        if k >= 1 and k < n:
            p = k / n
            E_M = N_sum * (-math.log(1 - p))
            term_hyper = (1 - n/N_sum) * (p * (1 - p)) / n
            term_binom = (p * (1 - p)) / N_sum
            var_p_total = term_hyper + term_binom
            term1 = (N_sum / (1 - p))**2 * var_p_total
            term2 = 0.5 * (N_sum / (1 - p)**2)**2 * var_p_total**2
            Var_M = term1 + term2
            CV = (Var_M**0.5) / E_M if E_M != 0 else 0
            CV_grid[i,j] = CV
        else:
            CV_grid[i,j] = 1e10  


contour = plt.contour(x_grid, n_grid, CV_grid, levels=[0.1], colors='black', linewidths=1)  
proxy = mlines.Line2D([], [], color='black', linewidth=0.1, label='CV=0.1')



plt.legend(handles=[proxy, line_kn, group2], loc='upper right', fontsize=20)
plt.xlabel('E(M)', fontsize=25, )
plt.ylabel('n', fontsize=25,) 
plt.title('Scatter Plot of CV (M) with E(M) and n for\nN = {}'.format(N_sum), fontsize=25, fontweight='bold')



ax.tick_params(axis='x', labelsize=20)  
ax.tick_params(axis='y', labelsize=20)  








cbar.ax.tick_params(labelsize=20)  







plt.tight_layout()


plt.savefig('2D图模拟.png')  


plt.show()