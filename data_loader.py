# This program selects a model and saves the results of simulations
import argparse
import json
from time import time
import taichi as ti
import numpy as np
from tqdm import tqdm

from models.srw import srw
from models.urw import urw
from models.percolation import largest_cluster_size as percolation

@ti.data_oriented
class DataLoader():
    def __init__(self, args):
        threads = args["threads"]
        # Initialize Taichi (CPU, parallel)
        ti.init(arch=ti.cpu, cpu_max_num_threads=threads)

        self.model = args["model"]
        self.rho = args["rho"]
        self.time_buget = args["time_budget"]
        self.savedir = args["save_directory"]

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model", 
            type=str,
            choices = ["percolation", "srw", "urw","rwre"],
            default = "srw",
            help = "Chooses a model to simulate, the options are \n "
            "'percoltion' - to bond percolation in 2 dimensions, \n "
            "'srw' - to Simple Random walk in 1 dimension, \n"
            " 'urw' - Uniform step random walk in 1 dimension, \n "
            "'rwre' - Random Walk in Random invironment, it is a biased random walk over a 1 dimension diffusion exclusion process"
        )
        parser.add_argument(
            "--rho",
            type=float,
            default=2,
        )
        parser.add_argument(
            "-tb",
            "--time_budget",
            type=float,
            default= 10,
            help= "Budget time, in seconds, approximally to total time to make the simulations"
        )
        parser.add_argument(
            "-save",
            "--save_directory",
            type=str,
            default="simulation_data",
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=8,
            help= "maximum number of threads for parallelization"
        )

    def save_metadata(self, time_taken, dom):
        metadata = {
            "model": self.model,
            "rho":self.rho,
            "time_budget": self.time_buget, 
            "time_taken": time_taken,
            "n_variation": dom,
        }
        metadata_path = self.model_dir + "/" + self.model + "/metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @ti.func
    def generate_data(self, model, n, numb_simul):
        data = np.zeros(numb_simul)
        for i  in range(numb_simul):
            data[i] = model(n)
        
        return data
    
    def generate_full_data(self, model, rho, k1, k2, numb_simul):
        assert k1 < k2, "[k1, k2] must be a valid interval, k1 < k2"

        start_time = time()
        full_data = np.empty((k2-k1+1, numb_simul))
        progress_bar = tqdm(range(k2-k1+1, numb_simul))

        for i,k in enumerate(progress_bar):
            n = pow(rho,k)
            full_data[i,:] = np.array(
                self.generate_data(n+1, numb_simul) # I am simulating random walks with an odd number of steps to avoid rw=0 that gives problems when take log
            )
        


    @ti.kernel
    def run(self):
        rho=2
        k1,k2 = 3, 6
        numb_simul = 10
        model = srw
        df = self.generate_full_data(model,rho, k1, k2, numb_simul)

        print(df.shape)

    
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    DataLoader.add_arguments(parser)
    args = vars(parser.parse_args()) # using default arguments

    data = DataLoader(args)
    data.run()