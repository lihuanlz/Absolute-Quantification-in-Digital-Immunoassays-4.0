


import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import math
from scipy.special import i0
import matplotlib.ticker as ticker
N = 2000000
Ms = 100000




error_limit =1

M_N_ratios = np.logspace(np.log10(1/N), np.log10(-math.log(1/N)), num=100)

n_values = np.linspace(1, N, num=100, dtype=int)

max_postive_beads = 0
Ms_postive_ratio = 1 - np.exp(-Ms/N)


results = []


for M_N_ratio in M_N_ratios:  
    C1 = (1-np.exp(-M_N_ratio)) / (1-np.exp(-Ms/N))
    

    C2 = M_N_ratio * N / Ms
    error = np.abs((C1 - C2) / C2)
    
    
    results.append([M_N_ratio, error])

    if error < error_limit:
        max_postive_beads = M_N_ratio * N
    else:
        break


with open('7_error.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['M/N ratio', 'error'])  
    for result in results:
        writer.writerow(result)

print('CSV file has been written with the 7_error.csv.')



results = []




        


        
        
        
        


        
        
        
        


        

        
    



    
    
    
        
        
        

        




for M_N_ratio in M_N_ratios:
    postive_ratio = 1 - np.exp(-M_N_ratio)
    for n in n_values:
        Ms_p = 1 - np.exp(-Ms/N)
        M_p = postive_ratio

        
        Ms_var = ((N**2) / ((1-Ms_p)**2)) * ((1-n/N) * Ms_p * (1-Ms_p) / n + Ms_p * (1-Ms_p) / N) + \
                 0.5 * ((N**2) / ((1-Ms_p)**4)) * ((1-n/N) * Ms_p * (1-Ms_p) / n + Ms_p * (1-Ms_p) / N)**2

        
        M_var = ((N**2) / ((1-M_p)**2)) * ((1-n/N) * M_p * (1-M_p) / n + M_p * (1-M_p) / N) + \
                0.5 * ((N**2) / ((1-M_p)**4)) * ((1-n/N) * M_p * (1-M_p) / n + M_p * (1-M_p) / N)**2

        Ms_mean = -math.log(1 - (Ms / N)) * N
        M_mean = -math.log(1 - postive_ratio) * N

        Ms_std = np.sqrt(Ms_var)
        M_std = np.sqrt(M_var)

        RSD = np.sqrt((M_std/M_mean)**2 + (Ms_std/Ms_mean)**2)
        
        if RSD < error_limit:
            results.append((M_N_ratio, n / N, RSD))



csv_filename = '7_RSD.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['M_N_ratio', 'n / N', 'RSD'])
    writer.writerows(results)

print(f'Results written to {csv_filename}')




error ={}
with open('7_error.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
        error[row[0]] = row[1]


with open('7_RSD.csv', 'r') as f_in, \
     open('7_updated_sampling_results1.csv', 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    
    header = next(reader)
    header.append('error')
    writer.writerow(header)

    
    for row in reader:
        if row[0] in error:
            row.append(error[row[0]])
        else:
            row.append('N/A')
        writer.writerow(row)

print("File '7_updated_sampling_results.csv' has been created.")





error ={}
with open('7_error.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
        error[row[0]] = row[1]


with open('7_RSD.csv', 'r') as f_in, \
     open('7_updated_sampling_results.csv', 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    
    header = next(reader)
    header.append('error')
    header.append('sum_of_last_two_columns')
    writer.writerow(header)

    
    for row in reader:
        if row[0] in error:
            row.append(error[row[0]])
            
            
            
            
            
            sum_of_last_two = math.sqrt(float(row[-2])**2 + float(row[-1])**2) 
            
            
            
            
            
            
            row.append(sum_of_last_two)
        else:
            row.append('N/A')
            row.append('N/A')
        writer.writerow(row)

print("File '7_updated_sampling_results.csv' has been created.")


data = []
with open('7_updated_sampling_results.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
     try:
        x_value = float(row[0])
        y_value = float(row[1])
        z_value = float(row[4]) if row[4] != 'N/A' else np.nan  
        data.append([x_value, y_value, z_value])
     except ValueError:
        
        print(f"Skipping row due to conversion error: {row}")


data = np.array(data)


x = data[:, 0]
y = data[:, 1]
z = data[:, 2]


X, Y = np.meshgrid(np.unique(x), np.unique(y))


Z = griddata((x, y), z, (X, Y), method='linear')




plt.figure(figsize=(10, 8))

scatter = plt.scatter(x, y, c=z.flatten(), cmap='tab10', s=20, alpha=1)

cbar = plt.colorbar(scatter)



ticks = np.linspace(np.nanmin(z), np.nanmax(z), 11)  
cbar.set_ticks(ticks)  




cbar.ax.tick_params(labelsize=15)  
cbar.set_label('Total Error', fontsize=15)  

cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


CS = plt.contour(X, Y, Z, levels=[0.05,0.1], colors='black', linewidths=2, fontsize=15)  
plt.clabel(CS, inline=True, fontsize=15, fmt='%1.2f')  

plt.xlabel('M/N ratio', fontsize=15)
plt.ylabel('n/N ratio', fontsize=15)
plt.title(f'Scatter Plot with Total Error\nN={N}, Ms={Ms}', fontsize=15)
plt.xscale('log')

plt.tight_layout()

plt.axvline(x=131/N, color='gray', linestyle='--', label='131/N')
plt.axvline(x=1.2040, color='black', linestyle='--', label='1.2040')



dpi_value = 600
plt.legend()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('figure S4D.svg', dpi=dpi_value)  


plt.show()
