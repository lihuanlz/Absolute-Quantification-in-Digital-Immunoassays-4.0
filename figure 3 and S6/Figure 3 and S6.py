


import numpy as np
import matplotlib.pyplot as plt
import csv  
import math

N = 2000000
aglost = 0.1
beadlost = 0.1
agleft = 1 - aglost
left = 1 - beadlost
blank = 100
n_values = range(1, 14)  


csv_data = []


plt.figure(figsize=(8, 7))


for n in n_values:
    Nn = N * left ** (n - 1)  
    M_max_n = -N * np.log(1 / N) * agleft ** (n - 1)  
    M_min_n = 10 * agleft ** (n - 1)  
    M_min_n_int = int(M_min_n)
    log_value = math.log10(M_min_n_int)
    if M_max_n > 0:
        M_values = np.logspace(log_value, np.log10(M_max_n), 20)  
        M_values = np.round(M_values).astype(int) 
    else:
        M_values = np.array([0.1])  
    
    
    lambda_values = M_values / Nn
    blank_term = blank / Nn
    K_values_modified = 1 - np.exp(- (lambda_values + blank_term))
    lambda_values_fromK = -np.log(1 - K_values_modified)

    
    for i in range(len(M_values)):
        csv_data.append([lambda_values[i], K_values_modified[i], Nn, n])

    
    plt.plot(lambda_values, K_values_modified, label=f'n={n-1}')
    plt.plot(lambda_values_fromK, K_values_modified)


n = 1
Nn1 = N
M_values_n1 = [1.2040 * N]
lambda_values_n1 = np.array(M_values_n1) / Nn1
blank_term1 = blank / Nn1
K_values_modified_n1 = 1 - np.exp(- (lambda_values_n1 + blank_term1))
plt.scatter(lambda_values_n1, K_values_modified_n1, color='red')
plt.annotate(f'λ={lambda_values_n1[0]:.4f}', (lambda_values_n1[0], K_values_modified_n1[0]),
              textcoords="offset points", xytext=(5, 20), ha='center', fontsize=15, color='red')



n = 13
Nn14 = N * left ** (n - 1)
agleft_factor = agleft ** (n - 1)
M_values_n14 = [(131 * agleft_factor), (1.2040 * N * agleft_factor)]
lambda_values_n14 = np.array(M_values_n14) / Nn14
blank_term14 = blank / Nn14
K_values_modified_n14 = 1 - np.exp(- (lambda_values_n14 + blank_term14))
plt.scatter(lambda_values_n14, K_values_modified_n14, color='blue')
for i in range(len(lambda_values_n14)):
    plt.annotate(f'λ={lambda_values_n14[i]:.6f}', (lambda_values_n14[i], K_values_modified_n14[i]),
                 textcoords="offset points", xytext=(50, -20), ha='center', fontsize=15, color='blue')


plt.xlabel('λ', fontsize=20)
plt.ylabel('K / N', fontsize=20)
plt.title(f'Relationship between λ and K / N\nInitial N={N}', fontsize=20)
plt.legend(title='n=iterations', fontsize=15)
plt.xscale('log')
plt.yscale('log')


plt.tick_params(axis='both', labelsize=20)  

ax = plt.gca()


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


ax.spines['left'].set_linewidth(2)  
ax.spines['bottom'].set_linewidth(2)  
plt.tight_layout()




plt.savefig('figure S5.SVG')
plt.show()




csv_data = []

header = ['M', 'λ (M / (N × left^(n-1)))', 'K / (N × left^(n-1))', 'N', 'wash times']

plt.figure(figsize=(8, 7))

for n in n_values:
    Nn = N * left ** (n - 1)
    M_max_n = -N * np.log(1 / N) * agleft ** (n - 1)
    M_min_n = 10 * agleft ** (n - 1)  
    M_min_n_int = int(M_min_n)
    log_value = math.log10(M_min_n_int)
    if M_max_n > 0:
        M_values = np.logspace(log_value, np.log10(M_max_n), 20)
        M_values = np.round(M_values).astype(int) 
    else:
        M_values = np.array([0.1])

    lambda_values = M_values / Nn
    blank_term = blank / Nn
    K_values_modified = 1 - np.exp(- (lambda_values + blank_term))
    
    
    for i in range(len(M_values)):
        csv_data.append([
            M_values[i],          
            lambda_values[i], 
            K_values_modified[i],
            Nn,
            n-1
        ])




csv_filename = 'wash_data.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  
    writer.writerows(csv_data)

print(f"Data has been saved to {csv_filename}")
