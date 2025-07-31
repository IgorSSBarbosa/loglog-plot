import numpy as np
import taichi as ti

'''  Defines a simple random walk, each step is +1 or -1
- k is the number of steps
- q is the probaility of an uppward step (+1)
- (1-q) is the probability of an downward step (-1)
'''

@ti.func
def srw(n: ti.i32, q: ti.f64=0.5) -> ti.f64:
    assert 0 < q and q < 1, "q must be in the open interval (0,1)"
    pos = 0.0
    for _ in range(n):
        pos += 1.0 if ti.random() < q else -1.0
    return pos

@ti.kernel
def main(n: ti.i32, q: ti.f64) -> float:
    data = srw(n,q)
    return data

if __name__=="__main__":
    # Example of usage

    ti.init(debug=True)
    n = 10**10
    q = 0.5 
    result = main(n,q)

    print(result)
