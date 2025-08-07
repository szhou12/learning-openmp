#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <list>
#include <utility>
#include <exception>
#include <vector>
#include <random>
#include <cstdlib>

using namespace std;

struct Result
{
	double timestamp;
	int threads;
};

// Function declarations
void initializeMatrix(vector<vector<double>>& matrix, int size, bool random = true);
void printMatrix(const vector<vector<double>>& matrix, int size, int maxDisplay = 5);
bool verifyResult(const vector<vector<double>>& c1, const vector<vector<double>>& c2, int size);
const Result blockedMatrixMultiplication(vector<vector<double>>& a, vector<vector<double>>& b, 
                                        vector<vector<double>>& c, int N, int NEIB, int nThreads);
const Result standardMatrixMultiplication(vector<vector<double>>& a, vector<vector<double>>& b, 
                                         vector<vector<double>>& c, int N, int nThreads);
const Result sequentialMatrixMultiplication(vector<vector<double>>& a, vector<vector<double>>& b, 
                                           vector<vector<double>>& c, int N);

int main(int argc, char* argv[])
{
	const short maxThreads = 16;  // Increased to support 16 threads
	int N, NEIB;
	short method;
	bool batchMode = false;
	int specificThreads = 0;

	// Check for command line arguments
	if (argc == 5) {
		N = atoi(argv[1]);
		NEIB = atoi(argv[2]);
		method = atoi(argv[3]);
		specificThreads = atoi(argv[4]);
		batchMode = true;
	}

	cout << fixed << setprecision(8) << endl;
	try
	{
		if (!batchMode) {
			while (true)
			{
				cout << "   Matrix size (N): "; cin >> N;
				cout << "   Block size (NEIB): "; cin >> NEIB;
				cout << "   Method (1 - blocked, 2 - standard, 3 - sequential): "; cin >> method;

				// Validate block size (only for blocked method)
				if (method == 1 && N % NEIB != 0)
				{
					cout << "   Error: Matrix size must be divisible by block size for blocked method!" << endl;
					continue;
				}
			}
		}

		// Common execution logic for both interactive and batch mode
		do {
			// Validate block size (only for blocked method)
			if (method == 1 && N % NEIB != 0)
			{
				if (batchMode) {
					cout << "Error: Matrix size must be divisible by block size for blocked method!" << endl;
					return 1;
				} else {
					cout << "   Error: Matrix size must be divisible by block size for blocked method!" << endl;
					continue;
				}
			}

			// Initialize matrices
			vector<vector<double>> a(N, vector<double>(N));
			vector<vector<double>> b(N, vector<double>(N));
			vector<vector<double>> c(N, vector<double>(N, 0.0));

			initializeMatrix(a, N);
			initializeMatrix(b, N);

			if (!batchMode) {
				cout << endl << "   Sample of matrix A (top-left corner):" << endl;
				printMatrix(a, N);
				cout << "   Sample of matrix B (top-left corner):" << endl;
				printMatrix(b, N);
			}

			list<pair<short, Result>> results;
			double sequentialTime = 0.0;
			
			if (batchMode && specificThreads > 0) {
				// Batch mode: run with specific thread count
				// Reset result matrix
				for (int row = 0; row < N; row++)
					for (int col = 0; col < N; col++)
						c[row][col] = 0.0;

				Result result;
				if (method == 3) {
					result = sequentialMatrixMultiplication(a, b, c, N);
				} else if (method == 1) {
					result = blockedMatrixMultiplication(a, b, c, N, NEIB, specificThreads);
				} else {
					result = standardMatrixMultiplication(a, b, c, N, specificThreads);
				}
				
				// Output in CSV format for Python parsing
				cout << method << "," << specificThreads << "," << fixed << setprecision(8) << result.timestamp << endl;
				return 0;
			}
			else {
				// Interactive mode: run full analysis
				// For sequential method, run only once
				if (method == 3)
				{
					// Reset result matrix
					for (int row = 0; row < N; row++)
						for (int col = 0; col < N; col++)
							c[row][col] = 0.0;

					Result result = sequentialMatrixMultiplication(a, b, c, N);
					sequentialTime = result.timestamp;
					pair<short, Result> s_result(1, result);
					results.push_back(s_result);
				}
				else
				{
					// For parallel methods, run with 1-maxThreads threads
					for (int i = 0; i < maxThreads; i++)
					{
						// Reset result matrix
						for (int row = 0; row < N; row++)
							for (int col = 0; col < N; col++)
								c[row][col] = 0.0;

						Result result = (method == 1) ?
							blockedMatrixMultiplication(a, b, c, N, NEIB, i + 1) :
							standardMatrixMultiplication(a, b, c, N, i + 1);

						// Store sequential baseline (1 thread) for speedup calculation
						if (i == 0) sequentialTime = result.timestamp;

						pair<short, Result> s_result(i + 1, result);
						results.push_back(s_result);
					}
				}
			}

			cout << endl << "   Results:" << endl;
			cout << "   Threads\tTime (sec)\tSpeedup\t\tEfficiency" << endl;
			cout << "   -------\t---------\t-------\t\t----------" << endl;
			
			for (auto & result : results)
			{
				double currentTime = result.second.timestamp;
				double speedup = sequentialTime / currentTime;
				double efficiency = speedup / result.first;
				
				cout << "   " << result.first 
					 << "\t\t" << fixed << setprecision(6) << currentTime
					 << "\t\t" << fixed << setprecision(2) << speedup
					 << "\t\t" << fixed << setprecision(2) << efficiency
					 << endl;
			}

			if (!batchMode) {
				cout << "   Sample of result matrix C (top-left corner):" << endl;
				printMatrix(c, N);
				cout << endl;
			}
		} while (!batchMode);
	}
	catch (exception & e)
	{
		cout << e.what() << endl;
	}
	if (!batchMode) cin.get();
	return 0;
}

const Result blockedMatrixMultiplication(vector<vector<double>>& a, vector<vector<double>>& b, 
                                        vector<vector<double>>& c, int N, int NEIB, int nThreads)
{
	const int NB = N / NEIB;  // Number of blocks per dimension
	double now = omp_get_wtime();
	
	int p, q, r, i, j, k;
	
	for (p = 0; p < NB; p++) {
		#pragma omp parallel for default(shared) private(q, r, i, j, k) num_threads(nThreads)
		for (q = 0; q < NB; q++)
			for (r = 0; r < NB; r++)
				for (i = p * NEIB; i < p * NEIB + NEIB; i++)
					for (j = q * NEIB; j < q * NEIB + NEIB; j++)
						for (k = r * NEIB; k < r * NEIB + NEIB; k++)
							c[i][j] = c[i][j] + a[i][k] * b[k][j];
	}
	
	return { omp_get_wtime() - now, nThreads };
}

const Result standardMatrixMultiplication(vector<vector<double>>& a, vector<vector<double>>& b, 
                                         vector<vector<double>>& c, int N, int nThreads)
{
	double now = omp_get_wtime();
	
	#pragma omp parallel for num_threads(nThreads)
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				c[i][j] += a[i][k] * b[k][j];
	
	return { omp_get_wtime() - now, nThreads };
}

const Result sequentialMatrixMultiplication(vector<vector<double>>& a, vector<vector<double>>& b, 
                                           vector<vector<double>>& c, int N)
{
	double now = omp_get_wtime();
	
	// Pure sequential - no OpenMP directives
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				c[i][j] += a[i][k] * b[k][j];
	
	return { omp_get_wtime() - now, 1 };
}

void initializeMatrix(vector<vector<double>>& matrix, int size, bool random)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dis(0.0, 10.0);
	
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (random)
				matrix[i][j] = dis(gen);
			else
				matrix[i][j] = i + j + 1;  // Simple pattern for testing
		}
	}
}

void printMatrix(const vector<vector<double>>& matrix, int size, int maxDisplay)
{
	int displaySize = min(size, maxDisplay);
	for (int i = 0; i < displaySize; i++)
	{
		cout << "   ";
		for (int j = 0; j < displaySize; j++)
		{
			cout << setw(8) << matrix[i][j] << " ";
		}
		if (size > maxDisplay) cout << "...";
		cout << endl;
	}
	if (size > maxDisplay)
	{
		cout << "   ..." << endl;
	}
	cout << endl;
}

bool verifyResult(const vector<vector<double>>& c1, const vector<vector<double>>& c2, int size)
{
	const double epsilon = 1e-9;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (abs(c1[i][j] - c2[i][j]) > epsilon)
				return false;
		}
	}
	return true;
}
