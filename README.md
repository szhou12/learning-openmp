# OpenMP Performance Analysis Project

A comprehensive study of OpenMP parallelization techniques with performance analysis, speedup measurements, and efficiency comparisons across different algorithms and thread configurations.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Matrix Multiplication](#matrix-multiplication)
- [Numerical Integration](#numerical-integration)
- [Performance Analysis](#performance-analysis)
- [Key Findings](#key-findings)
- [Setup and Usage](#setup-and-usage)
- [Results and Visualizations](#results-and-visualizations)

## 🎯 Overview

This project implements and analyzes the performance of parallel algorithms using OpenMP, comparing them against sequential implementations. We focus on two fundamental computational problems:

1. **Matrix Multiplication**: Demonstrating cache optimization benefits
2. **Numerical Integration**: Illustrating OpenMP overhead considerations

Each implementation includes:
- ✅ Sequential baseline implementations
- ✅ OpenMP parallel implementations  
- ✅ Comprehensive performance testing
- ✅ Automated speedup and efficiency analysis
- ✅ Visual performance comparison graphs

## 📁 Project Structure

```
learning-openmp/
├── blocked-matrix-multiplication.cpp    # Matrix multiplication with 3 methods
├── numerical-integration.cpp           # Numerical integration with 4 methods
├── performance_test.py                 # Matrix multiplication testing suite
├── integration_performance_test.py     # Numerical integration testing suite
├── Makefile                           # Build configuration
├── *.csv                             # Performance results data
└── *.png                             # Generated performance graphs
```

## 🔢 Matrix Multiplication

### Implemented Methods

#### **1. Blocked Matrix Multiplication (OpenMP)**
```cpp
for (p = 0; p < NB; p++) {
  #pragma omp parallel for default(shared) private(q, r, i, j, k) num_threads(nThreads)
  for (q = 0; q < NB; q++)
    for (r = 0; r < NB; r++)
      for (i = p * NEIB; i < p * NEIB + NEIB; i++)
        for (j = q * NEIB; j < q * NEIB + NEIB; j++)
          for (k = r * NEIB; k < r * NEIB + NEIB; k++)
            c[i][j] = c[i][j] + a[i][k] * b[k][j];
}
```
- **Cache-optimized**: Works on blocks that fit in CPU cache
- **Superior scaling**: Best performance for large matrices
- **Memory efficient**: Reduces memory bandwidth requirements

#### **2. Standard Matrix Multiplication (OpenMP)**
```cpp
#pragma omp parallel for num_threads(nThreads)
for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < N; k++)
      c[i][j] += a[i][k] * b[k][j];
```
- **Traditional approach**: Direct triple-nested loop parallelization
- **Row-wise parallelization**: Each thread processes different rows
- **Simple implementation**: Straightforward OpenMP usage

#### **3. Sequential Matrix Multiplication**
```cpp
// Pure sequential - no OpenMP directives
for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < N; k++)
      c[i][j] += a[i][k] * b[k][j];
```
- **Baseline implementation**: For speedup comparison
- **No parallelization overhead**: Single-threaded execution

### Performance Results (1024×1024 matrices)

| Method | 1 Thread | 2 Threads | 4 Threads | 8 Threads | 16 Threads |
|--------|----------|-----------|-----------|-----------|------------|
| **Sequential** | 1.808s (baseline) | - | - | - | - |
| **Blocked** | 1.604s (1.13×) | 0.891s (2.03×) | 0.596s (3.03×) | 0.418s (4.33×) | 0.416s (4.35×) |
| **Standard** | 1.808s (1.00×) | 1.176s (1.54×) | 0.683s (2.65×) | 0.735s (2.46×) | 0.649s (2.78×) |

**Key Insights:**
- 🏆 **Blocked method dominates**: Up to 4.35× speedup vs sequential
- 📈 **Excellent scaling**: Near-linear scaling up to 8 threads
- 🎯 **Cache optimization matters**: Blocked method 76% faster at 8 threads

## ∫ Numerical Integration

### Implemented Methods

#### **1. Rectangle Method (OpenMP)**
```cpp
#pragma omp parallel for num_threads(nThreads) reduction(+: s)
for (int i = 1; i <= N; i++) s += f(x1 + i * dx);
s *= dx;
```

#### **2. Trapezoidal Method (OpenMP)**
```cpp
#pragma omp parallel for num_threads(nThreads) reduction(+: s)
for (int i = 1; i < N; i++) s += f(x1 + i * dx);
s = (s + (f(x1) + f(x2)) / 2) * dx;
```

#### **3. Sequential Rectangle Method**
#### **4. Sequential Trapezoidal Method**
Pure sequential implementations without OpenMP directives.

### Performance Results (sin(x) from 0 to π, dx=0.0001)

| Method | 1 Thread | 2 Threads | 4 Threads | 8 Threads | 16 Threads |
|--------|----------|-----------|-----------|-----------|------------|
| **Rectangle Sequential** | 0.000097s | - | - | - | - |
| **Rectangle OpenMP** | 0.000175s (0.56×) | 0.000108s (0.90×) | 0.000116s (0.84×) | 0.000182s (0.53×) | 0.000403s (0.24×) |
| **Trapezoidal Sequential** | 0.000095s | - | - | - | - |
| **Trapezoidal OpenMP** | 0.000102s (0.93×) | 0.000076s (1.24×) | 0.000089s (1.06×) | 0.000153s (0.62×) | 0.000340s (0.28×) |

**Key Insights:**
- ⚠️ **OpenMP overhead dominates**: Sequential often faster for small problems
- 📉 **Poor scaling**: Performance degrades with more threads
- 🎯 **Problem size matters**: Computation too small to benefit from parallelization

## 📊 Performance Analysis

### Testing Framework Features

#### **Automated Testing Suite**
- **Multiple runs**: Averages over 3 runs for statistical accuracy
- **Thread scaling**: Tests with 1, 2, 4, 8, 16 threads
- **Batch mode**: Command-line interface for systematic testing
- **CSV output**: Machine-readable results for analysis

#### **Comprehensive Metrics**
- **Execution Time**: Wall-clock time measurement
- **Speedup**: `Sequential_Time / Parallel_Time`
- **Efficiency**: `Speedup / Number_of_Threads`
- **Accuracy**: Numerical result validation

#### **Visualization**
- **Speedup Graphs**: Performance scaling with thread count
- **Efficiency Plots**: Thread utilization analysis
- **Accuracy Comparison**: Numerical precision validation

### Key Performance Patterns

#### **Matrix Multiplication**
- ✅ **Benefits from parallelization**: Large computational workload
- ✅ **Cache optimization crucial**: Blocked algorithm significantly better
- ✅ **Good scaling**: Up to 8 threads show strong performance gains
- ⚠️ **Diminishing returns**: Efficiency drops beyond 8 threads

#### **Numerical Integration**
- ❌ **OpenMP overhead too high**: Small problem size
- ❌ **No scaling benefits**: Performance degrades with more threads
- ✅ **Perfect numerical accuracy**: All methods produce correct results
- 💡 **Best for sequential**: Simple problems better without parallelization

## 🔧 Setup and Usage

### Prerequisites
```bash
# macOS with Homebrew
brew install libomp

# Or ensure OpenMP support is available
```

### Compilation
```bash
# Build all programs
make

# Or build individually
make blocked-matrix-multiplication
make numerical-integration
```

### Running Tests

#### **Interactive Mode**
```bash
# Matrix multiplication
./blocked-matrix-multiplication

# Numerical integration  
./numerical-integration
```

#### **Automated Performance Testing**
```bash
# Run complete matrix multiplication analysis
python3 performance_test.py

# Run complete numerical integration analysis
python3 integration_performance_test.py
```

#### **Batch Mode Testing**
```bash
# Matrix multiplication: N NEIB METHOD THREADS
./blocked-matrix-multiplication 1024 128 1 8

# Integration: X1 X2 DX METHOD THREADS
./numerical-integration 0 3.14159 0.0001 1 8
```

### Method Parameters

#### **Matrix Multiplication**
- **Method 1**: Blocked parallel
- **Method 2**: Standard parallel  
- **Method 3**: Sequential

#### **Numerical Integration**
- **Method 1**: Rectangle parallel
- **Method 2**: Trapezoidal parallel
- **Method 3**: Rectangle sequential
- **Method 4**: Trapezoidal sequential

## 📈 Results and Visualizations

### Generated Files
- `performance_results.csv` - Matrix multiplication data
- `integration_results.csv` - Numerical integration data
- `speedup_graph.png` - Matrix multiplication speedup plots
- `efficiency_graph.png` - Matrix multiplication efficiency plots
- `integration_speedup_graph.png` - Integration speedup plots
- `integration_efficiency_graph.png` - Integration efficiency plots
- `integration_accuracy_graph.png` - Numerical accuracy comparison

## 🎯 Key Findings

### **When OpenMP Helps**
- ✅ **Large computational problems** (matrix multiplication)
- ✅ **Memory-intensive operations** with good cache optimization
- ✅ **Work that can be divided efficiently** among threads
- ✅ **Problems where computation >> parallelization overhead**

### **When OpenMP Hurts**
- ❌ **Small, fast computations** (numerical integration example)
- ❌ **Problems with high synchronization requirements**
- ❌ **Memory-bound operations** without cache optimization
- ❌ **Work that doesn't divide well** among threads

### **Optimization Insights**
- 🎯 **Algorithm choice matters**: Blocked matrix multiplication superior
- 📊 **Thread count optimization**: 4-8 threads often optimal
- 💾 **Cache effects dominant**: Memory access patterns crucial
- ⚖️ **Balance is key**: Overhead vs. computational benefit

### **Performance Recommendations**
1. **Profile before parallelizing**: Measure if OpenMP actually helps
2. **Optimize algorithms first**: Cache-friendly algorithms scale better
3. **Consider problem size**: Small problems may not benefit
4. **Test thread counts**: More threads ≠ better performance
5. **Measure efficiency**: Aim for high thread utilization

---

**Project demonstrates comprehensive OpenMP performance analysis with real-world insights into parallel computing trade-offs and optimization strategies.**