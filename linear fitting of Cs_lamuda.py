


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def model(lambda_, a, b):
    return a * (lambda_ - b)


lambda_data = np.array([ 
3.5161738318212505e-06,
7.55731785046044e-06,
1.6242954934712517e-05,
3.4911008142264036e-05,
7.503428374997192e-05,
0.0001612713020182114,
0.00034662065864870585,
0.0007449923172846668,
0.0016012131388154637,
0.0034414898736943498,
0.007396799503845273,
0.015897952604275158,
0.0341694941003049,
0.0734405464736887,
0.1578458800274106,
0.3392583938703946,
0.7291685902193494,
1.5672031777807438,
3.368392211885061,
7.239690586356927







], dtype=np.float64)
Cs_data = np.array([
1,
2.284486151375095,
5.218876975824593,
11.922452177001624,
27.236676888792058,
62.22181116192355,
142.14486591289077,
324.72797766706867,
741.8365679444594,
1694.7153660527467,
3871.5537842700737,
8844.511004468832,
20205.162905393692,
46158.41484364961,
105448.25947974424,
240895.088468083,
550321.4935396144,
1257201.8307953088,
2872060.171935295,
6561181.688702161



 

], dtype=np.float64)




popt, pcov = curve_fit(model, lambda_data, Cs_data, method='trf')


a_fitted, b_fitted = popt



Cs_fitted = model(lambda_data, a_fitted, b_fitted)


SS_tot = np.sum((Cs_data - np.mean(Cs_data))**2)


SS_res = np.sum((Cs_data - Cs_fitted)**2)


R2 = 1 - (SS_res / SS_tot)


print(f"拟合得到的方程形式: C_s(λ') = {a_fitted:.8f} * (λ' - {b_fitted:.8f})")
print(f"决定系数 R^2: {R2:.8f}")


lambda_fine = np.linspace(min(lambda_data), max(lambda_data), 100)  
Cs_fitted_fine = model(lambda_fine, a_fitted, b_fitted)  

plt.scatter(lambda_data, Cs_data, color='red', label='Data')
plt.plot(lambda_fine, Cs_fitted_fine, label='Fitted curve', color='blue')
plt.xlabel('λ\'')
plt.ylabel('C_s(λ\')')





plt.legend()
plt.show()
