import matplotlib.pyplot as plt
import numpy as np

def simple_plot_x_slices(x, t, value):
    for i in range(len(t)):
        time_value = round(float(t[i]),2)
        plt.figure(figsize=(12, 8))
        plt.plot(x, value[i, :], alpha=0.7, linewidth=1.0)
        plt.xlabel('x', fontsize=14)
        plt.ylabel(f"u(x,{time_value} с)", fontsize=14)
        plt.title(f"Распределение S_w на координатном промежутке", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"pictures/time_{time_value}.png")
        plt.close()

def plot_comparison(X_star, u_star, u_pred, save_path="pictures/temp"):
    Uniq_time_array = [X_star[0,1]]
    
    for x in X_star[:,1]:
        if x not in Uniq_time_array:
            Uniq_time_array.append(x)
    
    x_ticks = np.linspace(0,1,10)
    
    for i in range(1,len(Uniq_time_array[:-1]),10):
        S_array = []
        X_array = []
        S_true_array = []
        check_time = Uniq_time_array[i]
        count = 0
        
        t = X_star[0,1]
        while t != check_time:
            count = count + 1
            t = X_star[count,1]
        
        while check_time == Uniq_time_array[i]:
            X_array.append(X_star[count,0])
            S_array.append(u_pred[count])
            S_true_array.append(u_star[count])
            count = count + 1
            check_time = X_star[count,1]
            
        fig = plt.figure(figsize=(8,8))
        plt.plot(X_array,S_true_array,"-r", label = 'true values')
        plt.plot(X_array,S_array, color='navy', label = 'PINN predict', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('S')
        plt.legend()
        plt.grid()
        plt.xticks(x_ticks)
        
        filename = f'{save_path}/plot_{Uniq_time_array[i]:.3f}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()