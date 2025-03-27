





import numpy as np
from scipy.stats import hypergeom, binom, chi2
from scipy.optimize import minimize, root_scalar
from scipy.special import logsumexp
import pandas as pd


def log_likelihood(lambda_, N, n, k):
    K_min = k
    K_max = min(k + (N - n), N)
    if K_min > K_max:
        return -np.inf  
    K_values = np.arange(K_min, K_max + 1)
    log_hyper = hypergeom.logpmf(k, N, K_values, n)
    p = 1 - np.exp(-lambda_)
    log_binom = binom.logpmf(K_values, N, p)
    
    valid = np.isfinite(log_hyper) & np.isfinite(log_binom)
    if not np.any(valid):
        return -np.inf
    log_hyper_valid = log_hyper[valid]
    log_binom_valid = log_binom[valid]
    combined = log_hyper_valid + log_binom_valid
    return logsumexp(combined)


def neg_log_likelihood(lambda_, N, n, k):
    return -log_likelihood(lambda_, N, n, k)


def infer_lambda(N, n, k, initial_guess=0.1):
    try:
        result = minimize(
            neg_log_likelihood,
            x0=initial_guess,
            args=(N, n, k),
            bounds=[(1e-10, None)],  
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result.success:
            return result.x[0]
        else:
            
            result_tnc = minimize(
                neg_log_likelihood,
                x0=initial_guess,
                args=(N, n, k),
                bounds=[(1e-10, None)],
                method='TNC',
                options={'maxiter': 10000}
            )
            if result_tnc.success:
                return result_tnc.x[0]
            else:
                raise ValueError("优化失败: " + result_tnc.message)
    except Exception as e:
        raise ValueError(f"优化异常: {str(e)}")


def find_confidence_interval(N, n, k, mle_lambda, alpha=0.05):
    try:
        critical_value = chi2.ppf(1 - alpha, df=1) / 2
        target_log_likelihood = log_likelihood(mle_lambda, N, n, k) - critical_value

        def root_func(lambda_, target):
            return log_likelihood(lambda_, N, n, k) - target

        
        try:
            res_lower = root_scalar(
                lambda x: root_func(x, target_log_likelihood),
                bracket=[1e-10, mle_lambda],
                method='brentq'
            )
            lower = res_lower.root
        except:
            lower = 0

        
        try:
            res_upper = root_scalar(
                lambda x: root_func(x, target_log_likelihood),
                bracket=[mle_lambda, 10 * mle_lambda],
                method='brentq'
            )
            upper = res_upper.root
        except:
            upper = np.inf

        return (lower, upper)
    except:
        return (np.nan, np.nan)


def calculate_M_statistics(row, N_total):
    n_val = row['n']
    k_val = row['k']
    if n_val <= 0 or k_val < 0:
        return pd.Series({'lambda_mle': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan})
    
    
    try:
        if k_val == 0:
            initial_guess = 1e-8
        else:
            p_initial = min(k_val / n_val, 1 - 1e-10)
            initial_guess = -np.log(1 - p_initial)
    except:
        initial_guess = 0.1
    
    try:
        lambda_mle = infer_lambda(N_total, n_val, k_val, initial_guess=initial_guess)
        ci_lower, ci_upper = find_confidence_interval(N_total, n_val, k_val, lambda_mle)
    except:
        try:
            lambda_mle = infer_lambda(N_total, n_val, k_val, initial_guess=1.0)
            ci_lower, ci_upper = find_confidence_interval(N_total, n_val, k_val, lambda_mle)
        except:
            lambda_mle = np.nan
            ci_lower, ci_upper = (np.nan, np.nan)
    
    return pd.Series({'lambda_mle': lambda_mle, 'ci_lower': ci_lower, 'ci_upper': ci_upper})


if __name__ == "__main__":
    
    N_total = 500000  
    
    n = [33529,
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
    31775,
    ]    
    k = [123,
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
    124,
    ]

    
    if len(n) != len(k):
        raise ValueError("n和k的长度必须相同")
    
    df = pd.DataFrame({'n': n, 'k': k})
    
    
    results = df.apply(calculate_M_statistics, axis=1, N_total=N_total)
    
    
    final_df = pd.concat([df, results], axis=1)
    
    
    final_df.to_csv('results.csv', index=False)
    
    print("前5行计算结果:")
    print(final_df.head().round(4))