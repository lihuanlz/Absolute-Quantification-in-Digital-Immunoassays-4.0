import math
import pandas as pd
import numpy as np


N_sum = 500000























k_values = [123,
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
124




]
n_values = [
33529,
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
31775



]


if len(k_values) != len(n_values):
    raise ValueError("k_values and n_values must have the same length.")


def calculate_M_statistics_second_order(N, n, k, alpha=0.05):
    
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
        term_hyper = (1 - n / N) * (p * (1 - p)) / n  
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


results = []
for n, k in zip(n_values, k_values):
    stats = calculate_M_statistics_second_order(N_sum, n, k)
    results.append({'n': n, 'k': k, 'E_M': stats['E_M'], 'Var_M': stats['Var_M'], 'CV': stats['CV']})


df_results = pd.DataFrame(results)


df_results.to_csv('M_statistics_results.csv', index=False)


print(df_results)
