# This program selects a model and saves the results of simulations
import argparse
import json
import os
from time import localtime, strftime, time
import numpy as np
from tqdm import tqdm

from models.srw import srw
from models.urw import urw
from models.percolation import largest_cluster_size as perc
from meta_time_algorithm import simulation_manager

class DataLoader():
    def __init__(self, args):
        model_dict = {
            'srw': srw,
            'urw': urw,
            'percolation': perc,
        }
        self.model_name = args["model"]
        self.model = model_dict[self.model_name]
        self.rho = args["rho"]
        self.time_budget = args["time_budget"]
        self.savedir = args["save_directory"]
        self.J = args['j']
        self.pre_simulation_budget = args['pre_simulation_budget']

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
            default= 100,
            help= "Budget time, in seconds, approximally to total time to make the simulations"
        )
        parser.add_argument(
            "-save",
            "--save_directory",
            type=str,
            default="simulation_data",
        )
        parser.add_argument(
            "--numb_simul",
            "-ns",
            type=int,
            default=500,
        )
        parser.add_argument(
            "--k1",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--k2",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--j",
            type=int,
            default=10
        )
        parser.add_argument(
            "--pre_simulation_budget",
            type=float,
            default=0.01
        )

    def save_data(self,data, dom, time_taken, numb_simul):
        # Create a dictionary to save the data
        metadata = {
            "model": self.model_name,
            "rho":self.rho,
            "time_budget": self.time_budget, 
            "time_taken": time_taken,
            "n_variation": dom,
            "number_of_simulations": numb_simul,
            "results": data.tolist(),  # Convert ndarray to list
        }

        # create diretory
        # Get the local time and format it
        now = localtime()
        now_str = strftime("%m-%d-%Y_%H-%M-%S", now)

        out_put_path = self.savedir + "/" + self.model_name + now_str + "/"
        self.model_dir = out_put_path
        print(out_put_path)

        os.makedirs(out_put_path, exist_ok=True)

        metadata_path = out_put_path + "/metadata.json"
        print(f'save data in: {metadata_path}')
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def generate_data(self, model, n, numb_simul):
        data = np.zeros(numb_simul)
        pbar = tqdm(range(numb_simul),leave=False)
        pbar.set_description(f'Sampling for {n} size ...')

        for i  in pbar:
            data[i] = model(n)
        
        return data
    
    def generate_full_data(self, k1, k2, numb_simul):
        assert k1 < k2, "[k1, k2] must be a valid interval, k1 < k2"

        start_time = time()

        full_data = np.empty((k2-k1+1, numb_simul))
        progress_bar = tqdm(range(k1, k2+1),leave=False)
        progress_bar.set_description('Simulation progress')

        dom = []
        for i,k in enumerate(progress_bar):
            n = pow(self.rho,k)
            dom.append(n)
            full_data[i,:] = self.generate_data(self.model,n+1, numb_simul) # I am simulating random walks with an odd number of steps to avoid rw=0 that gives problems when take log
        
        end_time = time()
        time_taken = end_time - start_time
        return full_data, dom, time_taken


def main():
    # define the arguments parser
    parser = argparse.ArgumentParser(description="Initialization Arguments")
    DataLoader.add_arguments(parser)
    args = parser.parse_args() # Use empty list to avoid command line parsing

    simulator = DataLoader(vars(args))

    numb_simul, k2 = simulation_manager(simulator.model, simulator.time_budget, simulator.pre_simulation_budget)
    k1 = np.max([k2 - simulator.J,0])
    print(f'k1 : {k1}')

    data, dom, time_taken = simulator.generate_full_data(k1, k2, numb_simul)
    simulator.save_data(data, dom, time_taken, numb_simul) # saving as json the results


if __name__=='__main__':
    main()

    
        
