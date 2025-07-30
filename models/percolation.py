import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize Taichi (CPU, parallel)
ti.init(arch=ti.cpu, cpu_max_num_threads=8)

@ti.data_oriented
class BondPercolation:
    def __init__(self, n, p=0.5, periodic=False, dtype=ti.i32):
        """
        Initialize bond percolation simulation
        
        Args:
            n (int): grid size (n x n)
            p (float): probability of bond being open (0-1)
            periodic (bool): whether to use periodic boundary conditions
        """
        self.n = n
        self.p = p
        self.periodic = periodic
        
        # Horizontal bonds (n x n-1)
        self.h_bonds = ti.field(dtype=dtype, shape=(n, n-1))
        # Vertical bonds (n-1 x n)
        self.v_bonds = ti.field(dtype=dtype, shape=(n-1, n))
        
        # For periodic boundaries
        self.h_periodic = ti.field(dtype=dtype, shape=n)
        self.v_periodic = ti.field(dtype=dtype, shape=n)

        # Cluster labels and Union-Find structures
        self.parent = ti.field(dtype=dtype, shape=(n, n))
        self.size = ti.field(dtype=dtype, shape=(n, n))
        self.labels = ti.field(dtype=dtype, shape=(n, n))
    
    @ti.kernel
    def generate_bonds(self):
        """Generate random horizontal and vertical bonds"""
        for i, j in ti.ndrange(self.n, self.n-1):
            self.h_bonds[i, j] = 1 if ti.random() < self.p else 0
            
        for i, j in ti.ndrange(self.n-1, self.n):
            self.v_bonds[i, j] = 1 if ti.random() < self.p else 0
            
        if self.periodic:
            for i in range(self.n):
                self.h_periodic[i] = 1 if ti.random() < self.p else 0
                self.v_periodic[i] = 1 if ti.random() < self.p else 0
    
    @ti.func
    def find(self, i: ti.i32, j: ti.i32) -> ti.i32:
        """Find root with path compression"""
        root_i, root_j = i, j
        # encodes the parent as a single integer of the form root_i * n + root_j
        while (self.parent[root_i, root_j] != root_i * self.n + root_j):
            temp = self.parent[root_i, root_j]
            root_i, root_j = temp // self.n, temp % self.n # decodes the parent back to root_i, root_j
            
        # Path compression
        while (i != root_i or j != root_j):
            temp = self.parent[i, j]
            self.parent[i, j] = root_i * self.n + root_j
            i, j = temp // self.n, temp % self.n
            
        return root_i * self.n + root_j 
    
    @ti.func
    def union(self, i1: ti.i32, j1: ti.i32, i2: ti.i32, j2: ti.i32):
        """Union by size"""
        root1 = self.find(i1, j1)
        root2 = self.find(i2, j2)
        
        if root1 != root2:
            r1_i, r1_j = root1 // self.n, root1 % self.n # decodes the root back to r1_i, r1_j
            r2_i, r2_j = root2 // self.n, root2 % self.n # decodes the root back to r2_i, r2_j
            
            if self.size[r1_i, r1_j] < self.size[r2_i, r2_j]:
                self.parent[r1_i, r1_j] = root2
                self.size[r2_i, r2_j] += self.size[r1_i, r1_j]
            else:
                self.parent[r2_i, r2_j] = root1
                self.size[r1_i, r1_j] += self.size[r2_i, r2_j]
    
    @ti.kernel
    def find_clusters(self):
        """Find connected clusters using Union-Find"""
        # Initialize Union-Find
        for i, j in ti.ndrange(self.n, self.n):
            self.parent[i, j] = i * self.n + j # Encode parent as a single integer
            self.size[i, j] = 1
            self.labels[i, j] = -1  # Initialize labels
            
        # Process horizontal bonds
        for i, j in ti.ndrange(self.n, self.n-1):
            if self.h_bonds[i, j]:
                self.union(i, j, i, j+1)
                
        # Process vertical bonds
        for i, j in ti.ndrange(self.n-1, self.n):
            if self.v_bonds[i, j]:
                self.union(i, j, i+1, j)
                
        # Process periodic boundaries if needed
        if self.periodic:
            for i in range(self.n):
                if self.h_periodic[i]:
                    self.union(i, 0, i, self.n-1)
                if self.v_periodic[i]:
                    self.union(0, i, self.n-1, i)
        
        # Assign final labels
        for i, j in ti.ndrange(self.n, self.n):
            root = self.find(i, j)
            self.labels[i, j] = root
    
    def get_largest_cluster(self):
        """Identify the largest cluster"""
        labels_np = self.labels.to_numpy()
        unique, counts = np.unique(labels_np, return_counts=True)
        if len(unique) <= 1:  # Only background
            return None
        largest_label = unique[np.argmax(counts)]
        return (labels_np == largest_label)
    
    def visualize(self, highlight_largest=True):
        """Visualize the percolation configuration"""
        # Create directory if it doesn't exist
        os.makedirs('cluster/images', exist_ok=True)
        
        plt.figure(figsize=(8, 8))
        
        # Convert bonds to numpy for plotting
        h_bonds_np = self.h_bonds.to_numpy()
        v_bonds_np = self.v_bonds.to_numpy()
        
        # Plot all bonds
        for i in range(self.n):
            for j in range(self.n-1):
                if h_bonds_np[i, j]:
                    plt.plot([j, j+1], [i, i], 'k-', lw=1)
                    
        for i in range(self.n-1):
            for j in range(self.n):
                if v_bonds_np[i, j]:
                    plt.plot([j, j], [i, i+1], 'k-', lw=1)
        
        # Highlight largest cluster if requested
        if highlight_largest:
            largest_cluster = self.get_largest_cluster()
            if largest_cluster is not None:
                sites = np.argwhere(largest_cluster)
                
                # Highlight bonds in largest cluster
                for i, j in sites:
                    # Horizontal bonds
                    if j < self.n-1 and h_bonds_np[i, j] and largest_cluster[i, j+1]:
                        plt.plot([j, j+1], [i, i], 'r-', lw=2)
                    # Vertical bonds
                    if i < self.n-1 and v_bonds_np[i, j] and largest_cluster[i+1, j]:
                        plt.plot([j, j], [i, i+1], 'r-', lw=2)
                    # Periodic bonds
                    if self.periodic:
                        if j == 0 and self.h_periodic[i] and largest_cluster[i, self.n-1]:
                            plt.plot([-0.2, 0], [i, i], 'r-', lw=2)
                        if i == 0 and self.v_periodic[j] and largest_cluster[self.n-1, j]:
                            plt.plot([j, j], [-0.2, 0], 'r-', lw=2)
        
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.xlim(-0.5, self.n-0.5)
        plt.ylim(-0.5, self.n-0.5)
        plt.gca().set_aspect('equal')
        plt.title(f'Bond Percolation (n={self.n}, p={self.p})',fontsize=18)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'images/percolation_n={self.n}_p={self.p}.png', dpi=300)

    def __largest_cluster_size__(self):
        """Return the size of the largest cluster"""
        largest_cluster = self.get_largest_cluster()
        return 0 if largest_cluster is None else np.sum(largest_cluster)

# Example usage
if __name__ == "__main__":
    # Parameters
    n = 50  # Grid size 
    p = 0.50  # Bond probability
    periodic = False  # Boundary conditions
    
    # Create and run simulation
    percolation = BondPercolation(n, p, periodic=False)
    percolation.generate_bonds()
    percolation.find_clusters()
    
    # Print largest cluster size
    print(f"Largest cluster size: {percolation.__largest_cluster_size__()}")
    
    # Visualize
    percolation.visualize(highlight_largest=True)

else:
    def largest_cluster_size(n, p, periodic=False):
        """Return the size of the largest cluster"""
        percolation = BondPercolation(n, p, periodic)
        percolation.generate_bonds()
        percolation.find_clusters()
        return percolation.__largest_cluster_size__()