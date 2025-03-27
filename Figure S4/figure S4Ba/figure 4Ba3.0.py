


import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as ticker

filename = '5.5_relative_error_between_lam_and_Poscent.csv'
data = pd.read_csv(filename)
Ms = data['Ms']
M_N_ratio = data['M_N_ratio']
N = data['N']
relative_error = data['relative_error']

valid_idx = ~(Ms.isna() | M_N_ratio.isna() | N.isna() | relative_error.isna())
Ms = Ms[valid_idx]
M_N_ratio = M_N_ratio[valid_idx]
N = N[valid_idx]
relative_error = relative_error[valid_idx]

x = np.log10(M_N_ratio)
y = np.log10(N)
z = np.log10(Ms)

points = np.vstack((x, y, z)).T
values = relative_error

interpolator = LinearNDInterpolator(points, values)


xGrid, yGrid, zGrid = np.meshgrid(np.linspace(min(x), max(x), 5),
                                   np.linspace(min(y), max(y), 5),
                                   np.linspace(min(z), max(z), 5))

vGrid = interpolator(xGrid, yGrid, zGrid)


step_size = 10  
x_sampled = x[::step_size]
y_sampled = y[::step_size]
z_sampled = z[::step_size]
relative_error_sampled = relative_error[::step_size]


fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')


sc = ax.scatter(x_sampled, y_sampled, z_sampled, c=relative_error_sampled, cmap='tab10', s=10, alpha=1)


cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.1)
label_fontsize = 15
title_fontsize = 20
cbar_label_fontsize = 15
tick_fontsize = 15

cbar.set_label('Relative Error', fontsize=cbar_label_fontsize)


vmin, vmax = sc.get_clim()


ticks = np.linspace(vmin, vmax, 11)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
cbar.set_ticks(ticks)


ax.set_xlabel('log10(M/N)', labelpad=5, fontsize=label_fontsize)
ax.set_ylabel('log10(N)', labelpad=5, fontsize=label_fontsize)
ax.set_zlabel('log10(Ms)', labelpad=1, fontsize=label_fontsize)
ax.set_title('3D Scatter Plot with Colors Representing Relative Error', fontsize=title_fontsize)


cbar.ax.tick_params(labelsize=tick_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.tick_params(axis='z', which='major', labelsize=tick_fontsize)


ax.view_init(elev=30, azim=135)


fig.tight_layout()
dpi_value = 300
plt.savefig('figure 4Ba.png', dpi=dpi_value, bbox_inches='tight')
plt.show()
