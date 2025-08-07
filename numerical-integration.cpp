#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <list>
#include <utility>
#include <exception>
#include <cstdlib>

using namespace std;

struct Result
{
	double timestamp, area;
};

double f(const double x);
const Result rectangleMethod(const double, const double, const double, const int);
const Result trapezoidalMethod(const double, const double, const double, const int);
const Result sequentialRectangleMethod(const double, const double, const double);
const Result sequentialTrapezoidalMethod(const double, const double, const double);

int main(int argc, char* argv[])
{
	const short maxThreads = 16;  // Increased to support 16 threads
	short method;
	double x1, x2, dx;
	bool batchMode = false;
	int specificThreads = 0;

	// Check for command line arguments: x1 x2 dx method threads
	if (argc == 6) {
		x1 = atof(argv[1]);
		x2 = atof(argv[2]);
		dx = atof(argv[3]);
		method = atoi(argv[4]);
		specificThreads = atoi(argv[5]);
		batchMode = true;
	}

	cout << fixed << setprecision(8) << endl;
	try
	{
		if (!batchMode) {
			while (true)
			{
				cout << "   X1: "; cin >> x1;
				cout << "   X2: "; cin >> x2;
				cout << "   dx: "; cin >> dx;
				cout << "   Method (1 - rectangle, 2 - trapezoidal, 3 - sequential rectangle, 4 - sequential trapezoidal): "; cin >> method;
			}
		}
		
		// Common execution logic for both interactive and batch mode
		do {
			if (batchMode && specificThreads > 0) {
				// Batch mode: run with specific thread count
				Result result;
				if (method == 3) {
					result = sequentialRectangleMethod(x1, x2, dx);
				} else if (method == 4) {
					result = sequentialTrapezoidalMethod(x1, x2, dx);
				} else if (method == 1) {
					result = rectangleMethod(x1, x2, dx, specificThreads);
				} else {
					result = trapezoidalMethod(x1, x2, dx, specificThreads);
				}
				
				// Output in CSV format for Python parsing
				cout << method << "," << specificThreads << "," << fixed << setprecision(8) << result.timestamp << "," << result.area << endl;
				return 0;
			}
			else {
				// Interactive mode: run full analysis
				list<pair<short, Result>> results;
				
				if (method == 3 || method == 4) {
					// Sequential methods - run only once
					Result result = (method == 3) ?
						sequentialRectangleMethod(x1, x2, dx) :
						sequentialTrapezoidalMethod(x1, x2, dx);
					pair<short, Result> s_result(1, result);
					results.push_back(s_result);
				} else {
					// Parallel methods - run with 1-maxThreads threads
					for (int i = 0; i < maxThreads; i++)
					{
						Result result = (method == 1) ?
							rectangleMethod(x1, x2, dx, i + 1) :
							trapezoidalMethod(x1, x2, dx, i + 1);

						pair<short, Result> s_result(i + 1, result);
						results.push_back(s_result);
					}
				}

				cout << endl << "   Results:" << endl;
				for (auto & result : results)
				{
					cout << "   Threads: " << result.first;
					cout << ", timestamp: " << result.second.timestamp;
					cout << ", area: " << result.second.area << endl;
				}
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

const Result rectangleMethod(const double x1, const double x2, const double dx, const int nThreads)
{
	const int N = static_cast<int>((x2 - x1) / dx);
	double now = omp_get_wtime();
	double s = 0;

	#pragma omp parallel for num_threads(nThreads) reduction(+: s)
	for (int i = 1; i <= N; i++) s += f(x1 + i * dx);

	s *= dx;
	 
	return { omp_get_wtime() - now, s };
}

const Result trapezoidalMethod(const double x1, const double x2, const double dx, const int nThreads)
{
	const int N = static_cast<int>((x2 - x1) / dx);
	double now = omp_get_wtime();
	double s = 0;

	#pragma omp parallel for num_threads(nThreads) reduction(+: s)
	for (int i = 1; i < N; i++) s += f(x1 + i * dx);

	s = (s + (f(x1) + f(x2)) / 2) * dx;
	 
	return { omp_get_wtime() - now, s };
}

const Result sequentialRectangleMethod(const double x1, const double x2, const double dx)
{
	const int N = static_cast<int>((x2 - x1) / dx);
	double now = omp_get_wtime();
	double s = 0;

	// Pure sequential - no OpenMP directives
	for (int i = 1; i <= N; i++) s += f(x1 + i * dx);

	s *= dx;
	 
	return { omp_get_wtime() - now, s };
}

const Result sequentialTrapezoidalMethod(const double x1, const double x2, const double dx)
{
	const int N = static_cast<int>((x2 - x1) / dx);
	double now = omp_get_wtime();
	double s = 0;

	// Pure sequential - no OpenMP directives
	for (int i = 1; i < N; i++) s += f(x1 + i * dx);

	s = (s + (f(x1) + f(x2)) / 2) * dx;
	 
	return { omp_get_wtime() - now, s };
}

double f(const double x)
{
	return sin(x);
}
