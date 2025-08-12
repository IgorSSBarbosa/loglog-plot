import numpy as np

'''  Defines a simple random walk, each step is +1 or -1
- k is the number of steps
- q is the probaility of an uppward step (+1)
- (1-q) is the probability of an downward step (-1)
'''

def srw(k,rand_seed,q=0.5):
    # Set the seed for reproducibility
    np.random.seed(rand_seed)
    x = np.random.choice([-1,1],k, p=[1-q, q])
    rw = sum(x)
    return np.abs(rw)

if __name__=="__main__":
    ''' Example of usage '''
    k = 15
    seed=100
    print(srw(k,seed))
