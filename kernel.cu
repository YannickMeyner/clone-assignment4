#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <thread>

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
    // startet den Kernel mit 1024 Threads (jeder Thread verarbeitet dann genau 2 Elemente)
    compactPrimesNoBankConflictsKernel<<<1, 1024>>>(d_input, d_output);
}


// Prädikate und lokale Präfixsummen für jeden Block berehncnen
__global__ void multiBlockPrimeKernel(int *inputArray, int *scannedPredicates, int *predicates, int *blockTotals)
{
    __shared__ int sharedPredicates[2048];
    int threadIdx2x = threadIdx.x * 2;
    int globalIdx = blockIdx.x * 2048 + threadIdx2x;
    
    // Primzahlen?
    int prime1 = isPrimeCUDA(inputArray[globalIdx]);
    int prime2 = isPrimeCUDA(inputArray[globalIdx + 1]);
    
    sharedPredicates[threadIdx2x] = prime1;
    sharedPredicates[threadIdx2x + 1] = prime2;
    
    predicates[globalIdx] = prime1;
    predicates[globalIdx + 1] = prime2;
    
    __syncthreads();
    
    // führt die exklusive Scan-Operation auf den Prädikaten dieses Blocks durch
    exclusivePrefixScan(threadIdx.x, sharedPredicates, &blockTotals[blockIdx.x]);
    
    // speichert die gescannten Prädikate
    scannedPredicates[globalIdx] = sharedPredicates[threadIdx2x];
    scannedPredicates[globalIdx + 1] = sharedPredicates[threadIdx2x + 1];
}

// Block-Summen scannen
__global__ void scanBlockTotals(int *blockTotals)
{
    __shared__ int sharedTotals[2048];
    int idx = threadIdx.x * 2;
    
    // lädt die Block-Summen in den Shared Memory
    sharedTotals[idx] = blockTotals[idx];
    sharedTotals[idx + 1] = blockTotals[idx + 1];
    
    __syncthreads();
    
    int dummy;
    exclusivePrefixScan(threadIdx.x, sharedTotals, &dummy);
    
    // speichert die gescanten Block-Summen zurück
    blockTotals[idx] = sharedTotals[idx];
    blockTotals[idx + 1] = sharedTotals[idx + 1];
}

// Ergebnisse kombinieren, um die endgültige Ausgabe zu erstellen
__global__ void assembleOutput(int *inputArray, int *scannedPredicates, int *predicates, int *blockTotals, int *outputArray)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // nur verarbeiten wenn das Prädikat auch wahr ist (Zahl == Primzahl)
    if (predicates[globalIdx] == 1)
    {
        int outputPosition = scannedPredicates[globalIdx] + blockTotals[globalIdx / 2048];
        outputArray[outputPosition] = inputArray[globalIdx];
    }
}

// Wrapper-Funktion
void compactPrimesMultiBlockGPU(int *d_input, int *d_output, int size)
{
    int *d_scannedPredicates, *d_predicates, *d_blockTotals;
    gpuErrCheck(cudaMalloc(&d_scannedPredicates, size * sizeof(int)));
    gpuErrCheck(cudaMalloc(&d_predicates, size * sizeof(int)));
    gpuErrCheck(cudaMalloc(&d_blockTotals, 2048 * sizeof(int)));
    
    // Schritt 1: Prädikate und lokale Präfixsummen berehnen
    dim3 blockSize(1024); // jeder Thread verarbeitet 2 Elemente
    dim3 gridSize(size / 2048); // 2048 Elemente pro Block
    multiBlockPrimeKernel<<<gridSize, blockSize>>>(d_input, d_scannedPredicates, d_predicates, d_blockTotals);
    
    // Schritt 2: Block-Summen scannen
    scanBlockTotals<<<1, 1024>>>(d_blockTotals);
    
    // Schritt 3: endgültige Ausgabe erstellen
    dim3 finalBlockSize(256);
    dim3 finalGridSize(size / 256);
    assembleOutput<<<finalGridSize, finalBlockSize>>>(d_input, d_scannedPredicates, d_predicates, d_blockTotals, d_output);
    
    gpuErrCheck(cudaFree(d_scannedPredicates));
    gpuErrCheck(cudaFree(d_predicates));
    gpuErrCheck(cudaFree(d_blockTotals));
}

const int TOTAL_ITERATIONS = 256;
const int NUM_STREAMS = 8;
const int ITERS_PER_STREAM = TOTAL_ITERATIONS / NUM_STREAMS;

// Array mit zufälligen Zahlen und einem Seed-Offset initialisieren --> Seed wird für jede Iteration geädnert
void initRandomArrayWithSeed(const int size, int *array, int seedOffset)
{
    srand(1337 + seedOffset);
    for (int index = 0; index < size; ++index)
    {
        array[index] = rand();
    }
}

void processStreamData(int streamIdx)
{
    const int dataSize = 1 << 22; // 2048 * 2048 Elemente
    
    // CUDA-Stream erstellen
    cudaStream_t stream;
    gpuErrCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    cudaEvent_t startEvent, endEvent, copyToGpuDone, copyFromGpuStart;
    gpuErrCheck(cudaEventCreate(&startEvent));
    gpuErrCheck(cudaEventCreate(&endEvent));
    gpuErrCheck(cudaEventCreate(&copyToGpuDone));
    gpuErrCheck(cudaEventCreate(&copyFromGpuStart));
    
    int *d_blockSums, *d_scannedValues, *d_predicateValues, *d_inputData, *d_outputData;
    gpuErrCheck(cudaMallocAsync((void **)&d_blockSums, 2048 * sizeof(int), stream));
    gpuErrCheck(cudaMallocAsync((void **)&d_scannedValues, dataSize * sizeof(int), stream));
    gpuErrCheck(cudaMallocAsync((void **)&d_predicateValues, dataSize * sizeof(int), stream));
    gpuErrCheck(cudaMallocAsync((void **)&d_inputData, dataSize * sizeof(int), stream));
    gpuErrCheck(cudaMallocAsync((void **)&d_outputData, 220000 * sizeof(int), stream));
    
    int *hostInput = new int[dataSize];
    int *hostOutput_CPU = new int[dataSize];
    int *hostResult = new int[220000];
    
    for (int i = 0; i < ITERS_PER_STREAM; ++i)
    {
        // neue Daten mit unterschiedlichem Seed generieren
        auto genStart = chrono::high_resolution_clock::now();
        initRandomArrayWithSeed(dataSize, hostInput, (streamIdx * ITERS_PER_STREAM) + i);
        auto genEnd = chrono::high_resolution_clock::now();
        
        gpuErrCheck(cudaEventRecord(startEvent, stream));
        gpuErrCheck(cudaMemcpyAsync(d_inputData, hostInput, dataSize * sizeof(int), cudaMemcpyHostToDevice, stream));
        gpuErrCheck(cudaEventRecord(copyToGpuDone, stream));
        
        // 1.: Prädikate und lokale Präfixsummen berechnen
        dim3 blockSize(1024); //jeder Thread verarbeitet 2 Elemente
        dim3 gridSize(dataSize / 2048); // 2048 Elemente pro Block
        multiBlockPrimeKernel<<<gridSize, blockSize, 0, stream>>>(d_inputData, d_scannedValues, d_predicateValues, d_blockSums);
        
        // 2.: Block-Summen scannen
        scanBlockTotals<<<1, 1024, 0, stream>>>(d_blockSums);
        
        // 3.: endgültige Ausgabe erstellen
        dim3 finalBlockSize(256);
        dim3 finalGridSize(dataSize / 256);
        assembleOutput<<<finalGridSize, finalBlockSize, 0, stream>>>(d_inputData, d_scannedValues, d_predicateValues, d_blockSums, d_outputData);
        
        gpuErrCheck(cudaEventRecord(copyFromGpuStart, stream));
        gpuErrCheck(cudaMemcpyAsync(hostResult, d_outputData, 220000 * sizeof(int), cudaMemcpyDeviceToHost, stream));
        gpuErrCheck(cudaEventRecord(endEvent, stream));
        
        // CPU-Berechnung (parallel zur GPU-Berechnung)
        auto cpuStart = chrono::high_resolution_clock::now();
        int primes_found = compact_prime(hostInput, hostOutput_CPU, dataSize);
        auto cpuEnd = chrono::high_resolution_clock::now();
        
        gpuErrCheck(cudaEventSynchronize(endEvent));
        
        auto valStart = chrono::high_resolution_clock::now();
        compareResultVec(hostOutput_CPU, hostResult, primes_found);
        auto valEnd = chrono::high_resolution_clock::now();
        
        float totalTime, copyToTime, copyFromTime;
        gpuErrCheck(cudaEventElapsedTime(&totalTime, startEvent, endEvent));
        gpuErrCheck(cudaEventElapsedTime(&copyToTime, startEvent, copyToGpuDone));
        gpuErrCheck(cudaEventElapsedTime(&copyFromTime, copyFromGpuStart, endEvent));
        
        cout << "CPU: " << chrono::duration_cast<chrono::milliseconds>(cpuEnd - cpuStart).count() << "ms ";
        cout << "und GPU: " << totalTime << "ms ";
        cout << "für " << primes_found << " Primzahlen auf Stream/Iteration " << streamIdx << "/" << i << endl;
    }
    
    gpuErrCheck(cudaFreeAsync(d_inputData, stream));
    gpuErrCheck(cudaFreeAsync(d_outputData, stream));
    gpuErrCheck(cudaFreeAsync(d_blockSums, stream));
    gpuErrCheck(cudaFreeAsync(d_scannedValues, stream));
    gpuErrCheck(cudaFreeAsync(d_predicateValues, stream));
    
    gpuErrCheck(cudaEventDestroy(startEvent));
    gpuErrCheck(cudaEventDestroy(endEvent));
    gpuErrCheck(cudaEventDestroy(copyToGpuDone));
    gpuErrCheck(cudaEventDestroy(copyFromGpuStart));
    
    delete[] hostInput;
    delete[] hostOutput_CPU;
    delete[] hostResult;
    
    gpuErrCheck(cudaStreamSynchronize(stream));
    gpuErrCheck(cudaStreamDestroy(stream));
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

    // --- Multi-Block Implementation ---
    cout << "\nTesting multi-block implementation for all items..." << endl;
    const int fullSize = 1 << 22;

    int* fullHostVector = new int[fullSize];
    initRandomArray(fullSize, fullHostVector);

    int* fullHostOutput_CPU = new int[fullSize];
    auto cpuFullStart = std::chrono::high_resolution_clock::now();
    int fullFound_primes = compact_prime(fullHostVector, fullHostOutput_CPU, fullSize);
    auto cpuFullEnd = std::chrono::high_resolution_clock::now();
    auto cpuFullDuration = std::chrono::duration_cast<std::chrono::milliseconds>(cpuFullEnd - cpuFullStart);

    cout << "Found " << fullFound_primes << " prime numbers." << endl;
    cout << "CPU implementation took " << cpuFullDuration.count() << " ms" << endl;

    int *d_fullInput, *d_fullOutput;
    gpuErrCheck(cudaMalloc(&d_fullInput, fullSize * sizeof(int)));
    gpuErrCheck(cudaMalloc(&d_fullOutput, fullFound_primes * sizeof(int)));

    gpuErrCheck(cudaMemcpy(d_fullInput, fullHostVector, fullSize * sizeof(int), cudaMemcpyHostToDevice));

    int* fullHostOutput_GPU = new int[fullFound_primes];

    gpuErrCheck(cudaEventRecord(start));
    compactPrimesMultiBlockGPU(d_fullInput, d_fullOutput, fullSize);
    gpuErrCheck(cudaEventRecord(stop));
    gpuErrCheck(cudaEventSynchronize(stop));

    float gpuTimeMultiBlock = 0;
    gpuErrCheck(cudaEventElapsedTime(&gpuTimeMultiBlock, start, stop));

    gpuErrCheck(cudaMemcpy(fullHostOutput_GPU, d_fullOutput, fullFound_primes * sizeof(int), cudaMemcpyDeviceToHost));

    // Validate results
    cout << "Validating multi-block results..." << endl;
    compareResultVec(fullHostOutput_CPU, fullHostOutput_GPU, fullFound_primes);
    cout << "Multi-block GPU implementation took " << gpuTimeMultiBlock << " ms" << endl;
    cout << "Speedup vs CPU: " << float(cpuFullDuration.count()) / gpuTimeMultiBlock << "x" << endl;

    cout << "\nTesting parallele Stream-Implementation with " << TOTAL_ITERATIONS << " iterations..." << endl;

    std::thread streamThreads[NUM_STREAMS];
    auto totalStartTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_STREAMS; i++)
    {
        streamThreads[i] = std::thread(processStreamData, i);
    }

    for (int i = 0; i < NUM_STREAMS; i++)
    {
        streamThreads[i].join();
    }

    cudaDeviceSynchronize();

    auto totalEndTime = std::chrono::high_resolution_clock::now();
    cout << TOTAL_ITERATIONS << " Iterations completed in " 
        << chrono::duration_cast<chrono::milliseconds>(totalEndTime - totalStartTime).count() 
        << "ms" << endl;
    cout << "Acverage Time per iteration: " 
        << chrono::duration_cast<chrono::milliseconds>(totalEndTime - totalStartTime).count() / (float)TOTAL_ITERATIONS 
        << "ms" << endl;

    // ToDo: Implement compact pattern on GPU, you can use var found_primes
	// to allocate the right size or to only loop/check what is needed in 
	// comparing the results (in case of an in-place implementation).

    // Free memory on device & host
    delete[] hostVector;
    delete[] hostOutput_CPU;
    delete[] hostOutput_GPU;
    delete[] fullHostVector;
    delete[] fullHostOutput_CPU;
    delete[] fullHostOutput_GPU;
    gpuErrCheck(cudaFree(d_input));
    gpuErrCheck(cudaFree(d_output));
    gpuErrCheck(cudaEventDestroy(start));
    gpuErrCheck(cudaEventDestroy(stop));
    gpuErrCheck(cudaFree(d_fullInput));
    gpuErrCheck(cudaFree(d_fullOutput));
    
    return 0;
}