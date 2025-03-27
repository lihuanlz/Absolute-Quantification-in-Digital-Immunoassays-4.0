




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


k_list = [
    [10000, 0.001, 10000, 0.001, 10000, 0.001, 10000, 0.001],
    [10000, 0.0001, 10000, 0.0001, 10000, 0.0001, 10000, 0.0001],
    [10000, 0.00001, 10000, 0.00001, 10000, 0.00001, 10000, 0.00001],
    [100000, 0.001, 100000, 0.001, 100000, 0.001, 100000, 0.001],
    [100000, 0.0001, 100000, 0.0001, 100000, 0.0001, 100000, 0.0001],
    [100000, 0.00001, 100000, 0.00001, 100000, 0.00001, 100000, 0.00001],
    [1000000, 0.001, 1000000, 0.001, 1000000, 0.001, 1000000, 0.001],
    [1000000, 0.0001, 1000000, 0.0001, 1000000, 0.0001, 1000000, 0.0001],
    [1000000, 0.00001, 1000000, 0.00001, 1000000, 0.00001, 1000000, 0.00001]
]


x0 = [0, 0, 0]


t = np.linspace(0, 3600, 50000)  


def model(t, x, k):
    x1, x2, x3 = x
    k1, k2, k3, k4, k5, k6, k7, k8 = k
    
    
    x4 = 1E-18 - (x1 + x2 + x3)
    x5 = 6.66E-9 - (x1 + x2)
    x6 = 6.66E-9 - (x1 + x3)
    
    
    dx1dt = k3*x2*x6 + k7*x3*x5 - (k4*x1 + k8*x1)
    dx2dt = k1*x4*x5 + k4*x1 - (k2*x2 + k3*x2*x6)
    dx3dt = k5*x4*x6 + k8*x1 - (k6*x3 + k7*x3*x5)
    
    return [dx1dt, dx2dt, dx3dt]


plt.figure(figsize=(8, 8))
initial_x4 = 1E-18  
legend_info = []

for k in k_list:
    
    sol = solve_ivp(model, [t[0], t[-1]], x0, t_eval=t, args=(k,), rtol=1e-60, atol=1e-60)
    
    
    x1_percentage = (sol.y[0] / initial_x4) * 100
    
    
    plt.plot(sol.t / 60, x1_percentage, linewidth=2)  
    legend_info.append(f'kon={k[0]:.1e}, koff={k[1]:.1e}')

plt.xlabel('Time (min)', fontsize=15)  
plt.ylabel('Complex as % of initial Ag concentration', fontsize=15)
plt.title('Change of complex as Percentage of Initial Ag Concentration over Time', fontsize=15)
plt.legend(legend_info, loc='best', fontsize=10)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


dpi_value = 300
plt.savefig('figure S3A.svg', dpi=dpi_value)



plt.show()
