#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>

using namespace std;


/* This is our CUDA call wrapper, we will use in PAC.
*
*  Almost all CUDA calls should be wrapped with this makro.
*  Errors from these calls will be catched and printed on the console.
*  If an error appears, the program will terminate.
*
* Example: gpuErrCheck(cudaMalloc(&deviceA, N * sizeof(int)));
*          gpuErrCheck(cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice));
*/
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
		if (abort)
		{
			exit(code);
		}
	}
}

// initialisiere den array mit random numbers
void initRandomArray(const int size, int *array)
{
    srand(1337);
    for (int i = 0; i < size; ++i) {
        array[i] = rand();
    }
}


// Compare result vectors
int compareResultVec(int* vectorCPU, int* vectorGPU, int size)
{
	int error = 0;
	for (int i = 0; i < size; i++)
	{
		error += abs(vectorCPU[i] - vectorGPU[i]);
	}
	if (error == 0)
	{
		cout << "No errors. All good!" << endl;
		return 0;
	}
	else
	{
		cout << "Accumulated error: " << error << endl;
		return -1;
	}
}


// Very inefficient way to check if a number is prime
// Use Baillie–PSW primality test or ECPP if you want to do it right :)
bool isPrime(int n)
{
	if (n == 2 || n == 3)
		return true;

	if (n <= 1 || n % 2 == 0 || n % 3 == 0)
		return false;

	for (int i = 5; i * i <= n; i += 6)
	{
		if (n % i == 0 || n % (i + 2) == 0)
			return false;
	}

	return true;
}

// GPU version von isPrime-func
__device__ int isPrimeCUDA(int n)
{
    if (n == 2 || n == 3)
        return 1;
    if (n <= 1 || n % 2 == 0 || n % 3 == 0)
        return 0;
    for (int i = 5; i * i <= n; i += 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
            return 0;
    }
    return 1;
}

//CPU implementation of compact pattern, returns number of found prime values.
int compact_prime(int* input, int* output, int size)
{
	int current_pos = 0;
	for (int i = 0; i < size; ++i)
	{
		int foo = input[i];
		if (isPrime(foo)) {
			output[current_pos] = input[i];
			current_pos += 1;
		}
	}
	return current_pos;
}

// scan algorithm für exklusive prefix-sum
__device__ void exclusivePrefixScan(int idx, int *array, int *totalSum)
{
    const int SIZE = 2048;
    int offset = 1;
    
    // up-sweep phase --> reduction
    for (int d = SIZE/2; d > 0; d /= 2)
    {
        __syncthreads();
        
        if (idx < d)
        {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            array[bi] += array[ai];
        }
        
        offset *= 2;
    }
    
    if (idx == 0)
    {
        *totalSum = array[SIZE - 1];
        array[SIZE - 1] = 0;
    }
    
    // down-sweep phase
    for (int d = 1; d < SIZE; d *= 2)
    {
        offset /= 2;
        __syncthreads();
        
        if (idx < d)
        {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            
            int t = array[ai];
            array[ai] = array[bi];
            array[bi] += t;
        }
    }
    
    __syncthreads();
}

// CUDA-Kernel für das Compact-Pattern (2048 Elemente, 1 Thread-Block)
__global__ void compactPrimesKernel(int *input, int *output)
{
    __shared__ int predicates[2048];
    __shared__ int totalPrimes;
    
    int idx = threadIdx.x;
    
    // jeder Thread verarbeitet zwei Elemente
    int i1 = idx * 2;
    int i2 = idx * 2 + 1;
    
    // überprüfe, ob die Zahlen Primzahlen sind und speichere das Ergebnis
    int isPrime1 = isPrimeCUDA(input[i1]);
    int isPrime2 = isPrimeCUDA(input[i2]);
    
    // speichere Prädikate im Shared Memory
    predicates[i1] = isPrime1;
    predicates[i2] = isPrime2;
    
    __syncthreads();
    
    // führe exklusive Präfixsumme auf den Prädikaten durch
    exclusivePrefixScan(idx, predicates, &totalPrimes);
    
    // schreibe Ausgabe (nur wenn das Element eine Primzahl ist)
    if (isPrime1 == 1)
    {
        output[predicates[i1]] = input[i1];
    }
    
    if (isPrime2 == 1)
    {
        output[predicates[i2]] = input[i2];
    }
}

// Wrapper-Funktion zum Starten des Kernels
void compactPrimesGPU(int *d_input, int *d_output)
{
    // starte Kernel mit 1024 Threads (jeder Thread verarbeitet 2 Elemente)
    compactPrimesKernel<<<1, 1024>>>(d_input, d_output);
}

#define BANK_PADDING(i) ((i) + (i) / 32)

// Scan-Algorithmus mit Bank-Konflikt-Vermeidung
__device__ void exclusivePrefixScanPadded(int idx, int *array, int *totalSum)
{
    const int SIZE = 2048;
    int offset = 1;
    
    // up-sweep phase --> reduction mit Padding zur Vermeidung von Bank-Konflikten
    for (int d = SIZE/2; d > 0; d /= 2)
    {
        __syncthreads();
        
        if (idx < d)
        {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            array[BANK_PADDING(bi)] += array[BANK_PADDING(ai)];
        }
        
        offset *= 2;
    }
    
    if (idx == 0)
    {
        *totalSum = array[BANK_PADDING(SIZE - 1)];
        array[BANK_PADDING(SIZE - 1)] = 0;
    }
    
    // down-sweep phase mit Padding
    for (int d = 1; d < SIZE; d *= 2)
    {
        offset /= 2;
        __syncthreads();
        
        if (idx < d)
        {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            
            int t = array[BANK_PADDING(ai)];
            array[BANK_PADDING(ai)] = array[BANK_PADDING(bi)];
            array[BANK_PADDING(bi)] += t;
        }
    }
    
    __syncthreads();
}

// CUDA-Kernel für das Compact-Pattern mit Bank-Konflikt-Vermeidung
__global__ void compactPrimesNoBankConflictsKernel(int *input, int *output)
{
    // "gepaddeltes" Array für Shared Memory (2048 + Padding)
    __shared__ int predicatesPadded[2048 + 64]; // Zusätzliche 64 Elemente für Padding (2048/32)
    __shared__ int totalPrimes;
    
    int idx = threadIdx.x;
    
    // jeder Thread verarbeitet zwei Elemente
    int i1 = idx * 2;
    int i2 = idx * 2 + 1;
    
    // überprüft, ob die Zahlen Primzahlen sind und speicherz das Ergebnis
    int isPrime1 = isPrimeCUDA(input[i1]);
    int isPrime2 = isPrimeCUDA(input[i2]);
    
    // speichert Prädikate im gepaddelten Shared Memory
    predicatesPadded[BANK_PADDING(i1)] = isPrime1;
    predicatesPadded[BANK_PADDING(i2)] = isPrime2;
    
    __syncthreads();
    
    // führt exklusive Präfixsumme auf den Prädikaten durch (mit Bank-Konflikt-Vermeidung)
    exclusivePrefixScanPadded(idx, predicatesPadded, &totalPrimes);
    
    // schreibt in die Ausgabe (nur wenn das Element eine Primzahl ist)
    if (isPrime1 == 1)
    {
        output[predicatesPadded[BANK_PADDING(i1)]] = input[i1];
    }
    
    if (isPrime2 == 1)
    {
        output[predicatesPadded[BANK_PADDING(i2)]] = input[i2];
    }
}

// Wrapper-Funktion zum Starten des Kernels mit Bank-Konflikt-Vermeidung
void compactPrimesNoBankConflictsGPU(int *d_input, int *d_output)
{
    // startet Kernel mit 1024 Threads (jeder Thread verarbeitet 2 Elemente)
    compactPrimesNoBankConflictsKernel<<<1, 1024>>>(d_input, d_output);
}

int main(void)
{
    // Define the size of the vector
    // const int size = 1 << 22;
    const int size = 2048;
    // This gives you 2048 * 2048 items,
	// sounds like a perfect 2 stage fit for the sum scan implementation
    
    // Allocate and prepare input vector
    int* hostVector = new int[size];
    srand(1337);  // We have the same pseudo-random numbers each time
	for (int index = 0; index < size; ++index) {
		hostVector[index] = rand();
	}
    
    // Make things easy, so use a same sized output buffer
    int* hostOutput_CPU = new int[size];
    
    // --- CPU Implementation ---
    // Measure CPU time
    auto cpuStart = std::chrono::high_resolution_clock::now();
    int found_primes = compact_prime(hostVector, hostOutput_CPU, size);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart);
    
    cout << "Found " << found_primes << " prime numbers." << endl;
    cout << "CPU implementation took " << cpuDuration.count() << " ms" << endl;
    
    // --- GPU Implementations ---
    // Allocate GPU memory
    int *d_input, *d_output;
    gpuErrCheck(cudaMalloc(&d_input, size * sizeof(int)));
    gpuErrCheck(cudaMalloc(&d_output, found_primes * sizeof(int)));
    
    // Copy input to GPU
    gpuErrCheck(cudaMemcpy(d_input, hostVector, size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate host memory for GPU results
    int* hostOutput_GPU = new int[found_primes];
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    gpuErrCheck(cudaEventCreate(&start));
    gpuErrCheck(cudaEventCreate(&stop));
    
    // --- Basic Implementation ---
    // Measure GPU time for basic implementation
    gpuErrCheck(cudaEventRecord(start));
    compactPrimesGPU(d_input, d_output);
    gpuErrCheck(cudaEventRecord(stop));
    gpuErrCheck(cudaEventSynchronize(stop));
    
    float gpuTime = 0;
    gpuErrCheck(cudaEventElapsedTime(&gpuTime, start, stop));
    
    // Copy results back
    gpuErrCheck(cudaMemcpy(hostOutput_GPU, d_output, found_primes * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Validate results
    cout << "\nTesting basic implementation..." << endl;
    cout << "Validating results..." << endl;
    compareResultVec(hostOutput_CPU, hostOutput_GPU, found_primes);
    
    cout << "Basic GPU implementation took " << gpuTime << " ms" << endl;
    cout << "Speedup vs CPU: " << float(cpuDuration.count()) / gpuTime << "x" << endl;
    
    // --- Bank-Conflict-Free Implementation ---
    cout << "\nTesting bank-conflict-free implementation..." << endl;
    
    // Measure GPU time for bank-conflict-free implementation
    gpuErrCheck(cudaEventRecord(start));
    compactPrimesNoBankConflictsGPU(d_input, d_output);
    gpuErrCheck(cudaEventRecord(stop));
    gpuErrCheck(cudaEventSynchronize(stop));
    
    float gpuTimeNoBankConflicts = 0;
    gpuErrCheck(cudaEventElapsedTime(&gpuTimeNoBankConflicts, start, stop));
    
    // Copy results back
    gpuErrCheck(cudaMemcpy(hostOutput_GPU, d_output, found_primes * sizeof(int), cudaMemcpyDeviceToHost));
    
    
    // Validate results
    cout << "Validating results..." << endl;
    compareResultVec(hostOutput_CPU, hostOutput_GPU, found_primes);
    
    cout << "GPU implementation (no bank conflicts) took " << gpuTimeNoBankConflicts << " ms" << endl;
    cout << "Speedup vs CPU: " << float(cpuDuration.count()) / gpuTimeNoBankConflicts << "x" << endl;
    cout << "Improvement over basic implementation: " << gpuTime / gpuTimeNoBankConflicts << "x" << endl;

    // ToDo: Implement compact pattern on GPU, you can use var found_primes
	// to allocate the right size or to only loop/check what is needed in 
	// comparing the results (in case of an in-place implementation).

    // Free memory on device & host
    delete[] hostVector;
    delete[] hostOutput_CPU;
    delete[] hostOutput_GPU;
    gpuErrCheck(cudaFree(d_input));
    gpuErrCheck(cudaFree(d_output));
    gpuErrCheck(cudaEventDestroy(start));
    gpuErrCheck(cudaEventDestroy(stop));
    
    return 0;
}