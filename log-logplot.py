import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress

class LoglogPlotter():
    def __init__(self,args):
        self.simulation = args["data_path"]
        self.data_path = 'simulation/'+ self.simulation + '/metadata.json'
        self.basex = args["basex"]
        self.basey = args["basey"]

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--data_path",
            type=str,
            default='srw08-11-2025_19-20-57'
        )
        parser.add_argument(
            "--basex",
            type=float,
            default=4,
        )
        parser.add_argument(
            "--basey",
            type=float,
            default=4,
        )

    def compute_estimators(self,data):
        mean = np.mean(data, axis=1)
        var = np.var(data, axis=1, ddof=1)
        return mean, var

    def compute_critical_exponent(self, dom , X_n, plot=True, confidence_interval=False):
        """
        Compute the critical exponent (α) and R² from a power-law relationship X_n ~ n^α.
        
        Parameters:
        -----------
        n : array_like
            Array of system sizes (e.g., simulation sizes).
        X_n : array_like
            Array of measured values (should follow X_n ~ n^α).
        plot : bool, optional
            If True, generates a log-log plot (default: True).
        confidence_interval : bool, optional
            If True, prints confidence intervals, over mean points
        
        Returns:
        --------
        alpha : float
            The critical exponent (slope of log(X_n) vs log(n)).
        r_squared : float
            The R-squared value of the linear fit.
        """
        # Convert inputs to numpy arrays
        # n = np.asarray(n)
        X_n = np.asarray(X_n)
        
        # Linear regression in log space
        log_n = np.log(dom)
        log_X_n = np.log(X_n)
        slope, intercept, r_value, _, _ = linregress(log_n, log_X_n)
        alpha = slope
        r_squared = r_value ** 2
        basex = 4 # Base for logarithmic scale
        basey = 4

        # Plotting
        if plot:
            fig,ax = plt.subplots(figsize=(8, 6))
            # plot data and fit
            ax.plot(dom, X_n, color='#004A87', marker='o', label='Data')
            ax.plot(dom, np.exp(intercept) * (dom ** slope), 'r--', 
                    label=f'Fit: $X_n \\sim n^{{{alpha:.4f}}}$\n$R² = {r_squared:.4f}$')
            ax.set_xscale('log', base=basex)
            ax.set_yscale('log', base=basey)

            ax.set_xlabel('$n$ (log scale)')
            ax.set_ylabel('$S_n$ (log scale)')
            ax.set(title='Log-Log Plot of percolation largerst Cluster')
            ax.legend()

            ax.grid()
            ax.grid(which='minor', color="0.9")

            plt.savefig(f'images/plot_{self.simulation}.png')


        return alpha, r_squared
    
    def plot(self):
        with open(self.data_path, 'r') as f:
            json_file = json.load(f)
        data = json_file['results']
        dom = json_file['n_variation']
        X_n, var = self.compute_estimators(data)
        # The variance will be used to compute confidence intervals 
        alpha, r_squared = self.compute_critical_exponent(dom, X_n, plot=True)
        print(f'alpha = {alpha}')
        print(f'r² = {r_squared}')



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Initialization Arguments")
    LoglogPlotter.add_arguments(parser)
    args = parser.parse_args() # Use empty list to avoid command line parsing

    plotter = LoglogPlotter(vars(args))

    plotter.plot()

    