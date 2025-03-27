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


points_x1 = [123,
86,
799,
784,
1730,
2225,
3830,
6384,
13472,
13281,
25090,
24884,
22722,
22692,
28759,
28319,
6125,
7948,
26393,
26265,
1185,
1403,
7447,
8634,
2218,
2227,
2847,
2383,
6342,
6328,
2253,
2485,
24013,
21170,
7899,
7681,
18698,
18712,
7855,
8497,
16843,
16166,
19450,
19771,
19117,
19901,
23342,
24010,
17056,
17117,
21455,
19481,
24917,
24141,
24288,
22558,
23398,
26480,
23860,
16785,
12567,
16504,
15187,
15162,
6359,
6459,
5641,
7685,
26685,
31239,
24552,
27572,
22593,
22972,
11940,
11378,
2935,
5067,
1591,
1509,
642,
611,
112,
124,]
points_y1 = [33529,
29568,
30257,
31110,
27297,
33491,
23171,
35706,
30509,
31713,
31098,
30145,
22988,
22866,
28760,
28320,
29071,
33328,
28472,
28456,
20960,
24727,
25139,
27922,
22721,
22127,
24019,
20192,
22124,
21902,
25262,
27985,
26420,
23602,
27587,
26023,
18699,
18713,
23020,
24709,
16844,
16167,
20085,
20342,
19173,
19937,
23344,
24012,
17073,
17136,
24776,
21719,
24918,
24142,
24784,
22893,
23404,
26483,
23871,
16792,
24276,
31450,
22859,
22096,
20070,
21347,
21735,
28879,
26687,
31241,
25211,
28192,
30914,
31997,
30768,
29375,
19332,
33952,
27176,
25337,
28955,
29381,
27738,
31775,]


points_x2 = [5655,776,125,44046,32237,13018,880,17596,2577,3898,10288,5435,1869,13412,2880,]  
points_y2 = [90581,114680,139732,130564,136445,115343,165529,118828,172737,142637,197965,170980,195295,203242,171355,]  






points_y1_normalized = [y / N_sum for y in points_y1]
points_y2_normalized = [y / N_sum for y in points_y2]


def convert_x(x_list, y_list, N_sum):
    converted = []
    for x, y in zip(x_list, y_list):
        ratio = x / y
        term = -math.log(1 - ratio) * N_sum
        converted.append(round(term, 4))  
    return converted


converted_x1 = convert_x(points_x1, points_y1, N_sum)
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


group1 = plt.scatter(converted_x1, points_y1, color='blue', s=40, label='SIMOA')  
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


plt.legend(handles=[proxy, line_kn, group1, group2], loc='upper right', fontsize=15)

plt.xlabel('E(M)', fontsize=25, )
plt.ylabel('n', fontsize=25,) 
plt.title('Scatter Plot of CV (M) with E(M) and n for\nN = {}'.format(N_sum), fontsize=25)



ax.tick_params(axis='x', labelsize=20)  
ax.tick_params(axis='y', labelsize=20)  








cbar.ax.tick_params(labelsize=20)  







plt.tight_layout()


plt.savefig('2D图模拟.svg')  


plt.show()