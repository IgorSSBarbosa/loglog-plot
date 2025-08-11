import time
import numpy as np
from matplotlib import pyplot as plt

from models.srw import srw

def meta_algorithm_complexity(model,time_budget=10,numbsimul=10,k1=2,plot=False):

    start_time = time.time()
    t = []
    a = []
    k = k1 # k is the size of simulations
    while time.time() - start_time < (time_budget/2):
        a_aux = np.empty((numbsimul,1))
        t_aux = np.empty((numbsimul,1))
        for j in range(numbsimul):
            lap_time = time.time()
            a_aux[j]=model(pow(2,k))  # running the model
            time_time = time.time()- lap_time
            t_aux[j]=time_time
        a.append(a_aux)
        t.append(t_aux)
        k = k + 1
    final_time = time.time() - start_time
    
    
    # turning into a np.array and reshaping to the wright format
    t = np.array(t, dtype=np.float32)
    n,m,o = t.shape
    t = t.reshape((n,m))
    # print(t[-1])

    # making the log-log-plot
    mean_t = np.log(t.mean(1))/np.log(2)
    median_t =np.log( np.quantile(t,0.5,1))/np.log(2)
    two_points = [median_t[n-2],median_t[n-1]]
    
    domain = range(k1+n-2,k1+n)
    domain2 = range(k1,k1+n)

    coef = np.polyfit(domain,two_points,  1)
    
    # dimensional complexity of the model
    d = coef[0] 
    L = coef[1]
    poly_out = [coef[1] + coef[0]*x for x in domain2]

    print(f"estimated complexity, d = {d}, L = {L}")
    print(f"biggest size simulated K={k-1}")
    print(f"----- coeficient calculated in {final_time} seconds ------")
    
    # ploting the graphics
    if plot:
        plt.title("Log-log-plot time simulations")
        plt.plot(domain2,mean_t, marker = 'o')
        plt.plot(domain2,median_t,marker='o',color='red')
        plt.plot(domain2,poly_out)
        plt.plot(domain,two_points,marker='o',color='green')
        plt.legend(('mean time', 'median time','linear regression','two_points'))
        plt.savefig('images/plot_meta_algorithm.png')

    return d,k

def budget_unit(model,budget_time=10,numbsimul=10):
    start_time = time.time()

    d,k2 = meta_algorithm_complexity(model,budget_time/2,numbsimul)  #------------- re do it right ---------------------
    k1 = 2
    domain = range(k1,k2)
    sub_budget_time = budget_time/(2*(k2-k1))
    
    a=[] # store the results of the simulation
    simulation = [] # store how many simulations of each size were made
    for k in domain:
        lap_time = time.time()
        aux = []
        while time.time()-lap_time < sub_budget_time:
            aux.append(model(pow(2,k)))
        a.append(aux)
        simulation.append(len(aux)*(pow(2,d*k)))

    # Calculating the budget unit
    bd = np.median(simulation)/sub_budget_time
    print('budget unit=',bd)
    print(f"----- Budget unit calculated in {time.time()-start_time} seconds -----")

    plt.title('Budget Unit')
    plt.plot(domain,simulation, marker = 'o')
    plt.savefig('images/plot_budget_unity.png')
    
    return bd, d

def optimal_sizes(bd,total_budget,d):
    numb_simul = int(1/2*(pow(bd*total_budget, 2/(d+2))))
    k2 = int((1/(d+2))*np.log(bd*total_budget)/np.log(2))
    return numb_simul,k2

def simulation_manager(model,total_budget,budget_time=0.1):
    '''
    parameters
    model: function,
        Can be one of the functions in models folder:srw, urw, rwre, perc
    total_budget: time in seconds,
        Total total that can be used for simulate.
    Budget_time, float in (0,1),
        Fraction of total_budget used to determine k2. 

    '''

    error_margin = 0.2

    # first approximation of constants bd,d
    bd_aux,d_aux = budget_unit(model,1) # it runs for only one second
    numb_simul_aux,k2_aux = optimal_sizes(bd_aux, budget_time*total_budget, d_aux*(1+error_margin))
    print(f"numbsimul_aux={numb_simul_aux}, k2_aux={k2_aux}")

    numb_simul_aux = np.max(10, numb_simul_aux//100)

    # Better approximation of constants bd,d
    print(f"running with model={model}, total_budget = {budget_time*total_budget}, numbsimulations= {numb_simul_aux} ")
    bd, d = budget_unit(model,budget_time*total_budget,numbsimul=int(numb_simul_aux))

    # Calculating the "optimal" size of simulation k and Numbsimul
    numb_simul,k2 = optimal_sizes(bd,total_budget*(1-budget_time),d*(1+error_margin))
    print("number of simulations =",numb_simul)
    print("K2 = ",k2)
    return numb_simul, k2

if __name__ == '__main__':
    # Example of usage

    numb_simul, k2 = simulation_manager(srw, total_budget= 3600, budget_time=0.01)

