

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker


N_sum = 2000000  


N_values = np.logspace(2, np.log10(N_sum), num=50, endpoint=True)
N_values = N_values.astype(int)

results = []

for N in N_values:
    
    n_values = np.logspace(2, np.log10(N), num=50, endpoint=True)
    n_values = n_values.astype(int)
    n_values = n_values[(n_values >= 1) & (n_values <= N)]
    
    for n in n_values:
        
        if n - 1 < 1:
            k_values = np.array([0])
        else:
            k_values = np.logspace(2, np.log10(n-1), num=50, endpoint=True)
            k_values = k_values.astype(int)
            k_values = k_values[k_values < n]
        
        for k in k_values:
            results.append({'N': N, 'n': n, 'k': k})


df_results = pd.DataFrame(results)



print(df_results.head())


df = pd.DataFrame(results)


df.to_csv('n_k_leq_N_results.csv', index=False)

print("符合条件的结果已保存到 n_k_leq_N_results.csv 文件中。")




   


    






    





    



def calculate_M_statistics_second_order(N, n, k, alpha=0.05):
    
    
    nan_series = pd.Series({
        'E_M': np.nan,
        'Var_M': np.nan,
        'CV': np.nan
    })

    
    try:
        N = int(N)
        n = int(n)
        k = int(k)
    except:
        return nan_series

    
    if n <= 0 or k < 0 or k > n or N <= 0:
        return nan_series

    
    p = k / n
    if p >= 1.0:  
        return pd.Series({
            'E_M': np.inf,
            'Var_M': np.inf,
            'CV': np.nan
        })

    
    try:
        E_M = -N * math.log(1 - p)
    except:
        return nan_series

    
    try:
        
        term_hyper = (N - n) / (N - 1) * (p * (1 - p)) / n
        
        
        term_binom = (p * (1 - p)) / N
        
        
        var_p_total = term_hyper + term_binom
        
        
        term1 = (N**2 / (1 - p)**2) * var_p_total
        term2 = 0.5 * (N**2 / (1 - p)**4) * var_p_total**2
        Var_M = term1 + term2
        
        
        if Var_M < 0:
            Var_M = np.nan
            
    except:
        return nan_series

    
    try:
        CV = math.sqrt(Var_M) / E_M if E_M != 0 else np.inf
    except:
        CV = np.nan

    return pd.Series({
        'E_M': E_M,
        'Var_M': Var_M,
        'CV': CV
    })




   











data = pd.read_csv('n_k_leq_N_results.csv', usecols=['N', 'n', 'k'])



data[['E_M', 'Var_M', 'CV']] = data.apply(lambda row: calculate_M_statistics_second_order(row['N'], row['n'], row['k']), axis=1)


data.to_csv('n_k_leq_N_results.csv', index=False)

print("计算的期望值、方差和变异系数已添加到 n_k_leq_N_results.csv 文件中。")



data = pd.read_csv('n_k_leq_N_results.csv', usecols=['N', 'n', 'k', 'E_M', 'Var_M', 'CV'])


data['log_N'] = np.log10(data['N'])
data['log_n'] = np.log10(data['n'])
data['E_M'] = np.log10(data['E_M'])







data.to_csv('n_k_leq_N_results_with_logs.csv', index=False)
print("包含对数值的结果已保存到 n_k_leq_N_results_with_logs.csv 文件中。")


data = pd.read_csv('n_k_leq_N_results_with_logs.csv')




fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(data['log_N'], data['log_n'], data['E_M'], c=data['CV'], cmap='tab10',s=10, alpha=0.7)






cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.1)
num_ticks = 11
cbar_ticks = np.linspace(np.min(data['CV']), np.max(data['CV']), num_ticks)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

cbar.set_ticks(cbar_ticks)


label_fontsize = 25
title_fontsize = 25
cbar_label_fontsize = 25
tick_fontsize = 25  


ax.tick_params(axis='x', labelsize=20)  
ax.tick_params(axis='y', labelsize=20)  
ax.tick_params(axis='z', labelsize=20)  







cbar.ax.tick_params(labelsize=20)  




cbar.set_label('CV', fontsize=cbar_label_fontsize)


ax.set_xlabel(r'$\log_{10} N$', labelpad=15, fontsize=label_fontsize)
ax.set_ylabel(r'$\log_{10} n$',labelpad=15, fontsize=label_fontsize)
ax.set_zlabel(r'$\log_{10} E(M)$', labelpad=15, fontsize=label_fontsize)


ax.set_title(f'3D Scatter Plot of CV (M) with N, n, E(M)', fontsize=25, fontweight='bold')



fig.tight_layout()  

plt.savefig('3D图模拟.png')  


plt.show()















