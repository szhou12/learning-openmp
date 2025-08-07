#!/usr/bin/env python3
"""
Performance Testing Script for Matrix Multiplication
Tests blocked, standard, and sequential approaches with varying thread counts
Generates speedup graphs
"""

import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class MatrixMultiplicationTester:
    def __init__(self, matrix_size=512, block_size=64):
        """
        Initialize the tester with matrix and block sizes
        
        Args:
            matrix_size (int): Size of square matrices (N x N)
            block_size (int): Block size for blocked algorithm
        """
        self.matrix_size = matrix_size
        self.block_size = block_size
        self.executable = "./blocked-matrix-multiplication"
        self.thread_counts = [1, 2, 4, 8, 16]
        self.methods = {
            1: "Blocked",
            2: "Standard", 
            3: "Sequential"
        }
        self.results = []
        
        # Validate inputs
        if matrix_size % block_size != 0:
            raise ValueError(f"Matrix size ({matrix_size}) must be divisible by block size ({block_size})")
    
    def run_single_test(self, method, threads):
        """
        Run a single test configuration
        
        Args:
            method (int): 1=blocked, 2=standard, 3=sequential
            threads (int): Number of threads to use
            
        Returns:
            float: Execution time in seconds
        """
        # For sequential method, threads parameter is ignored but still required
        actual_threads = 1 if method == 3 else threads
        
        cmd = [self.executable, str(self.matrix_size), str(self.block_size), str(method), str(actual_threads)]
        
        try:
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"Error running {self.methods[method]} with {threads} threads:")
                print(result.stderr)
                return None
            
            # Parse CSV output: method,threads,time
            output_line = result.stdout.strip()
            parts = output_line.split(',')
            
            if len(parts) == 3:
                return float(parts[2])
            else:
                print(f"Unexpected output format: {output_line}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Test timed out: {self.methods[method]} with {threads} threads")
            return None
        except Exception as e:
            print(f"Error running test: {e}")
            return None
    
    def run_all_tests(self, runs_per_test=3):
        """
        Run all test configurations multiple times and average results
        
        Args:
            runs_per_test (int): Number of runs per configuration for averaging
        """
        print(f"Testing matrix multiplication performance")
        print(f"Matrix size: {self.matrix_size}x{self.matrix_size}")
        print(f"Block size: {self.block_size}x{self.block_size}")
        print(f"Thread counts: {self.thread_counts}")
        print(f"Runs per test: {runs_per_test}")
        print("-" * 50)
        
        self.results = []
        
        # Test sequential method once (baseline)
        print("Testing Sequential method...")
        seq_times = []
        for run in range(runs_per_test):
            exec_time = self.run_single_test(3, 1)
            if exec_time is not None:
                seq_times.append(exec_time)
        
        if seq_times:
            avg_seq_time = np.mean(seq_times)
            self.results.append({
                'Method': 'Sequential',
                'Threads': 1,
                'Time': avg_seq_time,
                'Speedup': 1.0,
                'Efficiency': 1.0
            })
            print(f"  Sequential: {avg_seq_time:.6f} seconds (baseline)")
        else:
            print("  Sequential: FAILED")
            return
        
        # Test parallel methods with different thread counts
        for method_id, method_name in [(1, "Blocked"), (2, "Standard")]:
            print(f"\nTesting {method_name} method...")
            
            for threads in self.thread_counts:
                times = []
                
                for run in range(runs_per_test):
                    exec_time = self.run_single_test(method_id, threads)
                    if exec_time is not None:
                        times.append(exec_time)
                
                if times:
                    avg_time = np.mean(times)
                    speedup = avg_seq_time / avg_time
                    efficiency = speedup / threads
                    
                    self.results.append({
                        'Method': method_name,
                        'Threads': threads,
                        'Time': avg_time,
                        'Speedup': speedup,
                        'Efficiency': efficiency
                    })
                    
                    print(f"  {threads:2d} threads: {avg_time:.6f}s, speedup: {speedup:.2f}x, efficiency: {efficiency:.2f}")
                else:
                    print(f"  {threads:2d} threads: FAILED")
    
    def save_results(self, filename="performance_results.csv"):
        """Save results to CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
            return df
        return None
    
    def plot_speedup(self, save_plot=True, filename="speedup_graph.png"):
        """Generate speedup graph"""
        if not self.results:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot speedup for parallel methods only (exclude Sequential)
        parallel_methods = [method for method in df['Method'].unique() if method != 'Sequential']
        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']
        
        for i, method in enumerate(parallel_methods):
            method_data = df[df['Method'] == method]
            
            plt.plot(method_data['Threads'], method_data['Speedup'], 
                    color=colors[i], marker=markers[i], linewidth=2, markersize=8,
                    label=f'{method} Method')
        
        # Add ideal speedup line
        max_threads = max(self.thread_counts)
        ideal_threads = range(1, max_threads + 1)
        plt.plot(ideal_threads, ideal_threads, 'k--', alpha=0.5, 
                label='Ideal Speedup', linewidth=1)
        
        # Customize the plot
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.title(f'Matrix Multiplication Speedup Comparison\n'
                 f'Matrix Size: {self.matrix_size}×{self.matrix_size}, '
                 f'Block Size: {self.block_size}×{self.block_size}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, max_threads)
        plt.ylim(0, max(max_threads, df['Speedup'].max() * 1.1))
        
        # Add annotations for efficiency
        for i, method in enumerate(parallel_methods):
            method_data = df[df['Method'] == method]
            max_thread_data = method_data[method_data['Threads'] == max_threads]
            if not max_thread_data.empty:
                efficiency = max_thread_data['Efficiency'].iloc[0]
                speedup = max_thread_data['Speedup'].iloc[0]
                plt.annotate(f'Eff: {efficiency:.2f}', 
                           xy=(max_threads, speedup), 
                           xytext=(max_threads - 1, speedup + 0.5),
                           fontsize=9, ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Speedup graph saved to {filename}")
        
        plt.show()
    
    def plot_efficiency(self, save_plot=True, filename="efficiency_graph.png"):
        """Generate efficiency graph"""
        if not self.results:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot efficiency for parallel methods only
        parallel_methods = ['Blocked', 'Standard']
        colors = ['blue', 'red']
        markers = ['o', 's']
        
        for i, method in enumerate(parallel_methods):
            method_data = df[df['Method'] == method]
            if not method_data.empty:
                plt.plot(method_data['Threads'], method_data['Efficiency'], 
                        color=colors[i], marker=markers[i], linewidth=2, markersize=8,
                        label=f'{method} Method')
        
        # Add ideal efficiency line (1.0)
        max_threads = max(self.thread_counts)
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, 
                   label='Ideal Efficiency')
        
        # Customize the plot
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Efficiency', fontsize=12)
        plt.title(f'Matrix Multiplication Efficiency Comparison\n'
                 f'Matrix Size: {self.matrix_size}×{self.matrix_size}, '
                 f'Block Size: {self.block_size}×{self.block_size}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, max_threads)
        plt.ylim(0, 1.2)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Efficiency graph saved to {filename}")
        
        plt.show()

def main():
    """Main function to run performance tests"""
    # Test configuration
    MATRIX_SIZE = 1024  # Large size for better performance differences
    BLOCK_SIZE = 128    # Good block size for cache efficiency (1024/128 = 8 blocks per dimension)
    RUNS_PER_TEST = 3   # Average over 3 runs for better accuracy
    
    print("Matrix Multiplication Performance Testing")
    print("=" * 50)
    
    # Check if executable exists
    if not os.path.exists("./blocked-matrix-multiplication"):
        print("Error: blocked-matrix-multiplication executable not found!")
        print("Please compile the program first with: make blocked-matrix-multiplication")
        return
    
    try:
        # Create tester and run tests
        tester = MatrixMultiplicationTester(MATRIX_SIZE, BLOCK_SIZE)
        tester.run_all_tests(RUNS_PER_TEST)
        
        # Save results
        df = tester.save_results()
        
        if df is not None:
            print("\nPerformance Summary:")
            print(df.to_string(index=False, float_format='%.6f'))
            
            # Generate plots
            print("\nGenerating plots...")
            tester.plot_speedup()
            tester.plot_efficiency()
            
            print("\nTesting completed successfully!")
        else:
            print("No results to display")
            
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
