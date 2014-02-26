#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <helper_cuda.h>

#include <algorithm>
#include <time.h>
#include <limits.h>

//#define RADIX 4294967296
#define RADIX 2147483658
#define numElements 1048576
#define numIterations 10

void 
sequentialSort(int *unsorted, int *sorted)
{
   int *count, *prefix;

  // count number of entries for each value
  count = (int *) malloc (RADIX*sizeof(int));
  for (unsigned int i=0; i<RADIX; i++) count[i]=0;
  for (int i=0; i<numElements; i++) {
    count[unsorted[i]]++;
  }

  // prefix sum of count
  prefix = (int *) malloc (RADIX*sizeof(int));
  prefix[0] = 0;
  for (unsigned int i=1; i<RADIX; i++) {
    prefix[i] = prefix[i-1] + count[i];
  }
  
  // generate result

  int curr = 0;
  for (unsigned int i=0; i<RADIX; i++) {
    for (int j=0; j<count[i]; j++) {
      sorted[curr++] = i;
    }
  }

}



int
main(int argc, char **argv)
{
  int *unsorted, *sorted;

  // initialize list.  Value in range 0..RADIX
  unsorted = (int *) malloc (numElements*sizeof(int));
  sorted = (int *) malloc (numElements*sizeof(int));
  for (int i=0; i<numElements; i++) {
    unsorted[i] = (int) (rand() % RADIX);
  }

  //initialize list for Thrust
  thrust::host_vector<int> h_keys(numElements);
  thrust::host_vector<int> h_keysSorted(numElements);
  for (int i = 0; i < (int)numElements; i++)
     h_keys[i] = unsorted[i];

  // SEQUENTIAL RUN
  cudaEvent_t seq_start_event, seq_stop_event;
  checkCudaErrors(cudaEventCreate(&seq_start_event));
  checkCudaErrors(cudaEventCreate(&seq_stop_event));
  checkCudaErrors(cudaEventRecord(seq_start_event, 0));

  // TODO: THIS TAKES A FEW MINUTES AND SHOULD BE COMMENTED OUT FOR TESTING
  (void) sequentialSort(unsorted,sorted);

  checkCudaErrors(cudaEventRecord(seq_stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(seq_stop_event));

  float seq_time = 0;
  checkCudaErrors(cudaEventElapsedTime(&seq_time, seq_start_event, seq_stop_event));
  seq_time /= 1.0e3f;
  printf("radixSort (SEQ), Throughput = %.4f KElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-3f * numElements / seq_time, seq_time, numElements);


  // THRUST IMPLEMENTATION
  // copy onto GPU
  thrust::device_vector<int> d_keys;
    
  cudaEvent_t start_event, stop_event;
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));

  float totalTime = 0;
  // run multiple iterations to compute an average sort time
  for (int i = 0; i < numIterations; i++) {
        // reset data before sort
        d_keys= h_keys;

        checkCudaErrors(cudaEventRecord(start_event, 0));

        thrust::sort(d_keys.begin(), d_keys.end());

        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));

        float time = 0;
        checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
        totalTime += time;
    }

    totalTime /= (1.0e3f * numIterations);
    printf("radixSort in THRUST, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-6f * numElements / totalTime, totalTime, numElements);

    getLastCudaError("after radixsort");

    // Get results back to host for correctness checking
    thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());

    getLastCudaError("copying results to host memory");

    // Check results
    bool bTestResult = thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

    if (bTestResult) printf("THRUST: VALID!\n");

    // COMPARE SEQUENTIAL WITH THRUST
   bTestResult = true;
   for (int i = 0; i < (int)numElements; i++) {
     if (h_keysSorted[i] != sorted[i]) {
       bTestResult = false;
       break;
     }
   }
   if (bTestResult) printf("SEQ: VALID!\n");

   // TODO: NOW ADD YOUR OWN CODE, TIME AND VALIDATE AGAINST SEQUENTIAL
}

