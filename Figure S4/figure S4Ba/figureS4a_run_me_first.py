

import pandas as pd
import numpy as np


N = 2000000

error_limit =0.15
Ms_values = np.logspace(np.log10(1), np.log10(-np.log(1/N)*N), num=100,dtype=int)

N_values = np.logspace(np.log10(1), np.log10(N), num=100,dtype=int)

M_N_ratios = np.logspace(np.log10(1/N), np.log10(-np.log(1/N)), num=100)


data_list = []


for N in N_values:
    
    for Ms in Ms_values:
        
        for M_N_ratio in M_N_ratios:
            C1 = (1 - np.exp(-M_N_ratio)) / (1 - np.exp(-Ms/N))
            C2 = M_N_ratio * N / Ms
            relative_error = np.abs((C1 - C2) / C2)
            
            if relative_error < error_limit:
                data_list.append({'Ms': Ms,'N': N,  'M_N_ratio': M_N_ratio, 'relative_error': relative_error})


results_df = pd.DataFrame(data_list)


results_df.to_csv('5.5_relative_error_between_lam_and_Poscent.csv', index=False)


