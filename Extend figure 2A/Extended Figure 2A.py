
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter


Nmax = 2000000
N_values = np.logspace(0, np.log10(Nmax), num=100)
K_values = np.logspace(0, np.log10(Nmax-1), num=100)
K, N = np.meshgrid(K_values, N_values)


valid_mask = K <= N



M = np.where(valid_mask, -N * np.log(np.clip(1 - K / N, 1e-10, 1)), np.nan)  


M_no_nan = M[~np.isnan(M)]


plt.figure(figsize=(10, 8))
scatter = plt.scatter(N, K, c=M, cmap='tab10', norm=LogNorm(), s=10)  


cbar = plt.colorbar(scatter)
cbar.set_label('M', fontsize=15)



ticks = np.logspace(np.log10(np.nanmin(M_no_nan)), np.log10(np.nanmax(M_no_nan)), num=11)
cbar.set_ticks(ticks)


formatter = ScalarFormatter()
formatter.set_scientific(False)
formatter.set_powerlimits((-1, 1))
cbar.ax.yaxis.set_major_formatter(formatter)


plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('N', fontsize=15)
plt.ylabel('K', fontsize=15)
plt.title(f'Relationship between M, N, and K\nInitial Nmax={Nmax}', fontsize=15)


dpi_value = 300
plt.savefig('Extended_Data_Figure_2A_scatter.png', dpi=dpi_value)
plt.show()




