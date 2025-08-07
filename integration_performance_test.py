#!/usr/bin/env python3
"""
Performance Testing Script for Numerical Integration
Tests rectangle, trapezoidal, and sequential approaches with varying thread counts
Generates speedup graphs
"""

import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class NumericalIntegrationTester:
    def __init__(self, x1=0, x2=3.14159, dx=0.0001):
        """
        Initialize the tester with integration parameters
        
        Args:
            x1 (float): Lower bound of integration
            x2 (float): Upper bound of integration  
            dx (float): Step size for integration
        """
        self.x1 = x1
        self.x2 = x2
        self.dx = dx
        self.executable = "./numerical-integration"
        self.thread_counts = [1, 2, 4, 8, 16]
        self.methods = {
            1: "Rectangle (OpenMP)",
            2: "Trapezoidal (OpenMP)",
            3: "Rectangle (Sequential)",
            4: "Trapezoidal (Sequential)"
        }
        self.results = []
        
        # Calculate expected result for validation
        self.expected_result = 2.0  # integral of sin(x) from 0 to π
        self.tolerance = 0.01  # 1% tolerance for numerical accuracy
    
    def run_single_test(self, method, threads):
        """
        Run a single test configuration
        
        Args:
            method (int): 1=rectangle, 2=trapezoidal, 3=seq_rectangle, 4=seq_trapezoidal
            threads (int): Number of threads to use
            
        Returns:
            tuple: (execution_time, area_result)
        """
        # For sequential methods, threads parameter is ignored but still required
        actual_threads = 1 if method >= 3 else threads
        
        cmd = [self.executable, str(self.x1), str(self.x2), str(self.dx), str(method), str(actual_threads)]
        
        try:
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Error running {self.methods[method]} with {threads} threads:")
                print(result.stderr)
                return None, None
            
            # Parse CSV output: method,threads,time,area
            output_line = result.stdout.strip()
            parts = output_line.split(',')
            
            if len(parts) == 4:
                exec_time = float(parts[2])
                area = float(parts[3])
                
                # Validate numerical accuracy
                error = abs(area - self.expected_result) / self.expected_result
                if error > self.tolerance:
                    print(f"Warning: Large numerical error ({error:.3f}) for {self.methods[method]} with {threads} threads")
                
                return exec_time, area
            else:
                print(f"Unexpected output format: {output_line}")
                return None, None
                
        except subprocess.TimeoutExpired:
            print(f"Test timed out: {self.methods[method]} with {threads} threads")
            return None, None
        except Exception as e:
            print(f"Error running test: {e}")
            return None, None
    
    def run_all_tests(self, runs_per_test=3):
        """
        Run all test configurations multiple times and average results
        
        Args:
            runs_per_test (int): Number of runs per configuration for averaging
        """
        print(f"Testing numerical integration performance")
        print(f"Integration bounds: [{self.x1}, {self.x2}]")
        print(f"Step size: {self.dx}")
        print(f"Expected result: {self.expected_result}")
        print(f"Thread counts: {self.thread_counts}")
        print(f"Runs per test: {runs_per_test}")
        print("-" * 50)
        
        self.results = []
        
        # Test sequential methods once (baseline)
        for method_id in [3, 4]:  # Sequential rectangle and trapezoidal
            method_name = self.methods[method_id]
            print(f"Testing {method_name}...")
            
            times = []
            areas = []
            for run in range(runs_per_test):
                exec_time, area = self.run_single_test(method_id, 1)
                if exec_time is not None:
                    times.append(exec_time)
                    areas.append(area)
            
            if times:
                avg_time = np.mean(times)
                avg_area = np.mean(areas)
                self.results.append({
                    'Method': method_name,
                    'Threads': 1,
                    'Time': avg_time,
                    'Area': avg_area,
                    'Speedup': 1.0,
                    'Efficiency': 1.0
                })
                print(f"  Result: {avg_time:.6f} seconds, area: {avg_area:.8f}")
            else:
                print(f"  {method_name}: FAILED")
        
        # Store sequential baselines for speedup calculation
        seq_rect_time = None
        seq_trap_time = None
        for result in self.results:
            if "Rectangle (Sequential)" in result['Method']:
                seq_rect_time = result['Time']
            elif "Trapezoidal (Sequential)" in result['Method']:
                seq_trap_time = result['Time']
        
        # Test parallel methods with different thread counts
        for method_id, method_name in [(1, "Rectangle (OpenMP)"), (2, "Trapezoidal (OpenMP)")]:
            print(f"\nTesting {method_name}...")
            
            # Choose appropriate sequential baseline
            baseline_time = seq_rect_time if method_id == 1 else seq_trap_time
            if baseline_time is None:
                print(f"  Error: No sequential baseline for {method_name}")
                continue
            
            for threads in self.thread_counts:
                times = []
                areas = []
                
                for run in range(runs_per_test):
                    exec_time, area = self.run_single_test(method_id, threads)
                    if exec_time is not None:
                        times.append(exec_time)
                        areas.append(area)
                
                if times:
                    avg_time = np.mean(times)
                    avg_area = np.mean(areas)
                    speedup = baseline_time / avg_time
                    efficiency = speedup / threads
                    
                    self.results.append({
                        'Method': method_name,
                        'Threads': threads,
                        'Time': avg_time,
                        'Area': avg_area,
                        'Speedup': speedup,
                        'Efficiency': efficiency
                    })
                    
                    print(f"  {threads:2d} threads: {avg_time:.6f}s, speedup: {speedup:.2f}x, efficiency: {efficiency:.2f}")
                else:
                    print(f"  {threads:2d} threads: FAILED")
    
    def save_results(self, filename="integration_results.csv"):
        """Save results to CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
            return df
        return None
    
    def plot_speedup(self, save_plot=True, filename="integration_speedup_graph.png"):
        """Generate speedup graph"""
        if not self.results:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot speedup for parallel methods only (exclude Sequential)
        parallel_methods = [method for method in df['Method'].unique() if 'OpenMP' in method]
        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']
        
        for i, method in enumerate(parallel_methods):
            method_data = df[df['Method'] == method]
            
            plt.plot(method_data['Threads'], method_data['Speedup'], 
                    color=colors[i], marker=markers[i], linewidth=2, markersize=8,
                    label=f'{method}')
        
        # Add ideal speedup line
        max_threads = max(self.thread_counts)
        ideal_threads = range(1, max_threads + 1)
        plt.plot(ideal_threads, ideal_threads, 'k--', alpha=0.5, 
                label='Ideal Speedup', linewidth=1)
        
        # Customize the plot
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.title(f'Numerical Integration Speedup Comparison\n'
                 f'Integration: sin(x) from {self.x1} to {self.x2:.3f}, '
                 f'Step size: {self.dx}', fontsize=14)
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
    
    def plot_efficiency(self, save_plot=True, filename="integration_efficiency_graph.png"):
        """Generate efficiency graph"""
        if not self.results:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot efficiency for parallel methods only
        parallel_methods = [method for method in df['Method'].unique() if 'OpenMP' in method]
        colors = ['blue', 'red']
        markers = ['o', 's']
        
        for i, method in enumerate(parallel_methods):
            method_data = df[df['Method'] == method]
            if not method_data.empty:
                plt.plot(method_data['Threads'], method_data['Efficiency'], 
                        color=colors[i], marker=markers[i], linewidth=2, markersize=8,
                        label=f'{method}')
        
        # Add ideal efficiency line (1.0)
        max_threads = max(self.thread_counts)
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, 
                   label='Ideal Efficiency')
        
        # Customize the plot
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Efficiency', fontsize=12)
        plt.title(f'Numerical Integration Efficiency Comparison\n'
                 f'Integration: sin(x) from {self.x1} to {self.x2:.3f}, '
                 f'Step size: {self.dx}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, max_threads)
        plt.ylim(0, 1.2)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Efficiency graph saved to {filename}")
        
        plt.show()
    
    def plot_accuracy_comparison(self, save_plot=True, filename="integration_accuracy_graph.png"):
        """Generate accuracy comparison graph"""
        if not self.results:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Calculate relative errors
        df['Relative_Error'] = abs(df['Area'] - self.expected_result) / self.expected_result * 100
        
        methods = df['Method'].unique()
        colors = ['blue', 'red', 'green', 'orange']
        markers = ['o', 's', '^', 'D']
        
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            if 'Sequential' in method:
                # Sequential methods: single point
                plt.scatter(1, method_data['Relative_Error'].iloc[0], 
                           color=colors[i], marker=markers[i], s=100, label=method)
            else:
                # Parallel methods: line plot
                plt.plot(method_data['Threads'], method_data['Relative_Error'], 
                        color=colors[i], marker=markers[i], linewidth=2, markersize=8,
                        label=method)
        
        # Customize the plot
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Relative Error (%)', fontsize=12)
        plt.title(f'Numerical Integration Accuracy Comparison\n'
                 f'Expected Result: {self.expected_result}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, max(self.thread_counts))
        plt.yscale('log')  # Log scale for better visualization of small errors
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Accuracy graph saved to {filename}")
        
        plt.show()

def main():
    """Main function to run performance tests"""
    # Test configuration
    X1 = 0.0
    X2 = 3.14159  # π
    DX = 0.0001   # Small step size for good accuracy and reasonable computation time
    RUNS_PER_TEST = 3  # Average over 3 runs for better accuracy
    
    print("Numerical Integration Performance Testing")
    print("=" * 50)
    
    # Check if executable exists
    if not os.path.exists("./numerical-integration"):
        print("Error: numerical-integration executable not found!")
        print("Please compile the program first with: make numerical-integration")
        return
    
    try:
        # Create tester and run tests
        tester = NumericalIntegrationTester(X1, X2, DX)
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
            tester.plot_accuracy_comparison()
            
            print("\nTesting completed successfully!")
        else:
            print("No results to display")
            
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
