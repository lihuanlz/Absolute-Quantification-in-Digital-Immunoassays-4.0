



import numpy as np
import csv
import math

N = 2000000  
error_up_limit = 0.15

n_values = np.linspace(1, N-1, num=50)
pos_Ms_values = np.logspace(np.log10(1), np.log10(N-1), num=50)      
postive_ratios = np.logspace(np.log10(1/N), np.log10(1-1/N), num=50)

results = []

for postive_ratio in postive_ratios:
    for n in n_values:
        for pos_Ms in pos_Ms_values:
            Ms = pos_Ms
            
            Ms_mean = -math.log(1 - pos_Ms / N)*N
            
            p_Ms = pos_Ms / N
            Ms_var = ((N**2) / ((1-p_Ms)**2)) * ((1-n/N) * p_Ms * (1-p_Ms) / n + p_Ms * (1-p_Ms) / N) + \
                     0.5 * ((N**2) / ((1-p_Ms)**4)) * ((1-n/N) * p_Ms * (1-p_Ms) / n + p_Ms * (1-p_Ms) / N)**2
            Ms_std = np.sqrt(Ms_var)
            
            M_mean = -math.log(1 - postive_ratio)*N
            
            M_var = ((N**2) / ((1-postive_ratio)**2)) * ((1-n/N) * postive_ratio * (1-postive_ratio) / n + postive_ratio * (1-postive_ratio) / N) + \
                    0.5 * ((N**2) / ((1-postive_ratio)**4)) * ((1-n/N) * postive_ratio * (1-postive_ratio) / n + postive_ratio * (1-postive_ratio) / N)**2
            M_std = np.sqrt(M_var)
            
            if Ms_mean == 0 or M_mean == 0:
                relative_error = np.nan
            else:
                relative_error = np.sqrt((M_std/M_mean)**2 + (Ms_std/Ms_mean)**2)         
            
            if relative_error < error_up_limit:
                lamuda = -math.log(1 - postive_ratio)
                results.append((Ms, lamuda, n / N, relative_error))

csv_filename = '8_Ms_n_N_M_N_data.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Ms', 'lamuda', 'n_N_ratio', 'relative_error'])
    writer.writerows(results)
