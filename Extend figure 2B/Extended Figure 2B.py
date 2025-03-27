
import numpy as np
import matplotlib.pyplot as plt



N = 1000000
aglost = 0.1
beadlost = 0.05
M_max = -N * np.log(1 / N)
K_max = N * (1 - np.exp(-M_max / N))
left = 1 - beadlost
agleft=1- aglost

n_values = range(1, 13)  


plt.figure(figsize=(8, 8))


for n in n_values:
    
    M_max_n = -N * np.log(1 / N) * agleft**(n-1)
    if M_max_n > 0:
        M_values = np.logspace(np.log10(0.1), np.log10(M_max_n), 10000)  
    else:
        M_values = np.array([0.1])  
    K_values = N * left**(n-1) * (1 - np.exp((-M_values) / ((N * left**(n-1)))))  
    plt.plot(M_values/(N * left**(n-1)), K_values/(N * left**(n-1)), label=f'n={n}')  


n = 1
M_values_n1 = [131,1.2040 * N]
for M in M_values_n1:
    K_value = N * left**(n-1) * (1 - np.exp(-M / (N * left**(n-1))))  
    plt.scatter(M/(N * left**(n-1)), K_value/(N * left**(n-1)), color='red')  
    plt.annotate(f'M={M:.0f}', (M/N * left**(n-1), K_value/N * left**(n-1)), textcoords="offset points", xytext=(-30,10), ha='center', fontsize=15, color='red')


n = 13
M_values_n12 = [int(131 * agleft**(n-1)), int(1.2040 * N * agleft**(n-1))]
for M in M_values_n12:
    K_value = N * left**(n-1) * (1 - np.exp(-M / (N * left**(n-1))))  
    plt.scatter(M/(N * left**(n-1)), K_value/(N * left**(n-1)), color='blue')  
    plt.annotate(f'M={M:.0f}', (M/N * left**(n-1), K_value/N * left**(n-1)), textcoords="offset points", xytext=(30,-10), ha='center', fontsize=15, color='blue')


plt.xlabel('λ', fontsize=15)
plt.ylabel('K/N', fontsize=15)
plt.title(f'Relationship between M and K\n Initial N={N}', fontsize=15)
plt.legend(title='Number of Iterations', fontsize=15)

plt.xscale('log')
plt.yscale('log')


dpi_value=300
plt.savefig('Extended Data Figure 2B.png', dpi=dpi_value)  
plt.show()





