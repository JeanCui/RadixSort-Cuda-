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
//#define numElements 1048576
#define numElements 2048
#define numIterations 10

#define BLOCKSIZE 32
#define BINNUM 16
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

// for 2 bins
__device__ int getBit(unsigned int &num, int pos)
{
  return (num >> (pos)) & (0x1);
}

// for 16 bins
__device__ int getBinIndex(unsigned int &num, int pos)
{ 
  return (num >> (pos)) & (0xf);
}

__device__ void reductionSum(int *s_data, int *sum, int tx)
{
 // printf("jc\n");
  //int tx = threadIdx.x;

  //__shared__ int tile[2][BLOCKSIZE];

  //tile[0][tx] = s_data[tx];
  //tile[1][tx] = s_data[BLOCKSIZE+tx];

//  for(int i=0;i<BLOCKSIZE;i++)
 // {
 //   printf("%d\n",s_data[i]);
 // }

  // blocksize must be power of 2
  for(int i = BLOCKSIZE; i > 0; i = i >> 1)
  {
    if(tx < i)
    {
      s_data[tx] += s_data[i+tx];
     // s_data[tx+BLOCKSIZE] += s_data[BLOCKSIZE+i+tx];
    }
      __syncthreads();
  }

  if(tx == 0)
  {
    sum[0] = s_data[0];
   // sum[1] = s_data[BLOCKSIZE];
  }
  

}
__global__ void global_radixsort(unsigned int *d_keys,
    int pos, int *blockCount, int numBlocks)
{
  int tx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tx;

  // may be no need to use shared memory
 // __shared__ unsigned int tile[BLOCKSIZE];
 // tile[tx] = d_keys[i];

  __shared__ int perBlockCount[BINNUM*BLOCKSIZE];

  for(int i=0;i<BINNUM;++i)
    perBlockCount[i*BLOCKSIZE+tx] = 0;
  
  __syncthreads();
  
  int binIdx = getBinIndex(d_keys[idx], pos);

  perBlockCount[binIdx*BLOCKSIZE+tx] = 1;
  //printf("%d,%d\n",perBlockCount[zoo*BLOCKSIZE+tx],perBlockCount[(1-zoo)*BLOCKSIZE+tx]);
  __syncthreads();

  __shared__ int bin[BINNUM][BLOCKSIZE];
    
  for(int i=0;i<BINNUM;++i)
  {
    bin[i][tx] = perBlockCount[tx+i*BLOCKSIZE];
  }
  __syncthreads();
  
  for(int i = BLOCKSIZE/2; i > 0; i >>= 1)
  {
    if(tx < i)
    {
      for(int i=0;i<BINNUM;++i)
      {
        bin[i][tx] = bin[i][tx]+ bin[i][i+tx];
      }
    }
     __syncthreads();
  }

  if(tx == 0)
  {
    for(int i=0;i<BINNUM;++i)
    {
      blockCount[blockIdx.x+i*numBlocks] = bin[i][0];
    }
    //printf("Block %d, bin0--%d, bin1--%d\n",blockIdx.x,
    //  blockCount[blockIdx.x],blockCount[blockIdx.x+numBlocks]);
  }
}

__global__ void combineBlockCount(int *s_data, int *g_data)
{

  int tx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tx;
  
  extern __shared__ int data[];
  data[tx] = s_data[idx];
  __syncthreads();

  for(int i=blockDim.x/2; i > 0; i >>= 1)
  {
    if(tx < i)
      data[tx] += data[i+tx];

    __syncthreads();
  }

  if(tx == 0)
  {
    g_data[blockIdx.x] = data[0];
  }
}

__global__ void prefixSumBottomUp(int *s_data, int *g_data)
{
  int tx = threadIdx.x;
  int idx = blockIdx.x*blockDim.x+tx;

  extern  __shared__ int tile[];
  tile[tx] = s_data[idx];
  //tile[tx+1] = s_data[idx+1];

  int data_size = blockDim.x;

  //tile[2*tx+1] += tile[2*tx];
  // bottom-up phase
  for(int i=2, size=data_size/2; i <= data_size; i <<= 1, size>>=1)
  {
    if( tx < size)
      tile[(i*tx+i-1)] += tile[i*tx+i>>1-1];
    __syncthreads();
  }
  tile[data_size-1] = 0;

  __syncthreads();

  g_data[tx] = tile[tx];
 // printf("tx:%d, %d\n", tx, g_data[tx]);

}
__global__ void prefixSumTopDown(int *s_data, int *g_data)
{
  int tx = threadIdx.x;
  int idx = blockIdx.x*blockDim.x+tx;

  extern  __shared__ int tile[]; 
  tile[tx] = s_data[idx];

  int data_size = blockDim.x;
  
  // top-down phase
  for(int i=data_size,size=1; i>=2; i>>=1,size<<=1)
  {
    if(tx<size)
    {
      int tmp = tile[i*tx+i-1];
      //printf("i>>1:%d, i*tx:%d, i*tx+i>>1-1:%d\n", i>>1, i*tx, 0+1-1);
      tile[i*tx+i-1] += tile[i*tx+(i>>1)-1];
      tile[i*tx+(i>>1)-1] = tmp;
     // printf("tile[%d]:%d,tile[%d]:%d\n", i*tx+i-1, tile[i*tx+i-1], i*tx+i>>1-1,
     //     tile[i*tx+i>>1-1]);
    }
     __syncthreads();
  }

  g_data[tx] = tile[tx];
  //printf("tx:%d, %d\n", tx, g_data[tx]);
}

__global__ void scatter(unsigned int *s_data, unsigned int *g_data, int *prefixSum,
    int pos)
{
  int tx = threadIdx.x;
  //int idx = blockIdx.x*blockDim.x + tx;

  int idx = prefixSum[tx];
  //extern  __shared__ int count[]; // size = BINNUM

  //if(tx < binsize)
  //{
  //  count[tx] = prefixSum[tx];
  //  //printf("tx:%d,%d\n", tx, count[tx]);
  //}

  //__syncthreads();

  for (int i=0;i<numElements;++i)
  {

    unsigned int key = s_data[i];
    int bit = getBit(key, pos);
    if(tx == bit)
    {
      g_data[idx] = s_data[i];
      idx++;
    }
  }

  //printf("out idx:%d\n", out_idx);
  //count[out_idx]++; //atomic
}
__host__ void host_radixsort(unsigned int *h_keys, unsigned int *h_sorted)
{
  unsigned int *d_keys;
  //unsigned int *d_keysSorted;
  unsigned int numbytes = numElements*sizeof(unsigned int);

  checkCudaErrors(cudaMalloc((void **) &d_keys, numbytes));
  checkCudaErrors(cudaMemcpy(d_keys, h_keys, numbytes, cudaMemcpyHostToDevice));
 // checkCudaErrors(cudaMalloc((void**)&d_keysSorted, numbytes));
 // checkCudaErrors(cudaMemset(d_keysSorted, 0, numbytes));

  //int perBlockCount[2*BLOCKSIZE];
  int numBlocks = numElements / BLOCKSIZE;
 
  cudaEvent_t my_start_event, my_stop_event;
  checkCudaErrors(cudaEventCreate(&my_start_event));
  checkCudaErrors(cudaEventCreate(&my_stop_event));
 
  checkCudaErrors(cudaEventRecord(my_start_event, 0));
  int *blockCount;
  checkCudaErrors(cudaMalloc((void**)&blockCount, 2*numBlocks*sizeof(int)));
  int *overallCount;
  checkCudaErrors(cudaMalloc((void**)&overallCount, BINNUM*sizeof(int)));
  int *bottomUpResult;
  checkCudaErrors(cudaMalloc((void**)&bottomUpResult, BINNUM*sizeof(int)));
  int *prefixSumArray;
  checkCudaErrors(cudaMalloc((void**)&prefixSumArray, BINNUM*sizeof(int)));

  unsigned int *keys_tmp;
  checkCudaErrors(cudaMalloc((void**)&keys_tmp, numElements*sizeof(int)));

  for(int i = 0; i < 32; i+=4)
  {
    global_radixsort<<<numBlocks, BLOCKSIZE>>>(d_keys, i, blockCount, numBlocks);
    // combine per block counting
    //printf("jc\n");
    combineBlockCount<<<BINNUM, numBlocks, numBlocks*sizeof(int)>>>(blockCount, overallCount);
    // prefix sum
    
    prefixSumBottomUp<<<1, BINNUM, BINNUM*sizeof(int)>>>(overallCount,
        bottomUpResult);

    prefixSumTopDown<<<1, BINNUM, BINNUM*sizeof(int)>>>(bottomUpResult,
        prefixSumArray);
    //scatter to d_keys
    scatter<<<1, BINNUM>>>(d_keys, keys_tmp, prefixSumArray,
        i);
    checkCudaErrors(cudaMemcpy(d_keys, keys_tmp, numElements*sizeof(int),
          cudaMemcpyDeviceToDevice));

  }

  checkCudaErrors(cudaEventRecord(my_stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(my_stop_event));
  float my_time = 0;
  checkCudaErrors(cudaEventElapsedTime(&my_time, my_start_event, my_stop_event));
  my_time /= 1.0e3f;
  printf("radixSort (MyTest), Throughput = %.4f KElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-3f * numElements / my_time, my_time, numElements);

#define testreorder

#ifdef testblockcount
  int *hostBlockCount;
  hostBlockCount = (int*)malloc(2*numBlocks*sizeof(int));
  checkCudaErrors(cudaMemcpy(hostBlockCount, blockCount,
        2*numBlocks*sizeof(int), cudaMemcpyDeviceToHost));
  for(int i = 0;i<numBlocks;++i)
    printf("block %d: bin0:%d, bin1:%d\n",i, hostBlockCount[i],
      hostBlockCount[i+numBlocks]);
#endif
  
#ifdef testoverallcount
  int *hostOverallCount;
  hostOverallCount = (int*)malloc(BINNUM*sizeof(int));
  checkCudaErrors(cudaMemcpy(hostOverallCount, overallCount,
        BINNUM*sizeof(int), cudaMemcpyDeviceToHost));

  for(int i=0; i < BINNUM;++i)
    printf("bin%d, %d\n", i, hostOverallCount[i]);
#endif

#ifdef testprefixsum
  int *hostPrefixSum;
  hostPrefixSum = (int*)malloc(BINNUM*sizeof(int));
  checkCudaErrors(cudaMemcpy(hostPrefixSum, prefixSumArray,
        BINNUM*sizeof(int), cudaMemcpyDeviceToHost));
  
  for(int i=0; i < BINNUM;++i)
    printf("prefix%d, %d\n", i, hostPrefixSum[i]);
#endif

#ifdef testreorder
  unsigned int *hostKeys;
  hostKeys = (unsigned int*)malloc(numElements*sizeof(int));
  checkCudaErrors(cudaMemcpy(hostKeys, d_keys,
        numElements*sizeof(int), cudaMemcpyDeviceToHost));
  
  for(int i=0; i < numElements;++i)
    printf("%d\n",hostKeys[i]);
#endif



  checkCudaErrors(cudaDeviceSynchronize());
  //checkCudaErrors(cudaMemcpy(d_keysSorted, h_sorted, numbytes, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_keys));
  //checkCudaErrors(cudaFree(d_keysSorted));
  



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
    //unsorted[i] = 1;
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
  //(voir) sequentialSort(unsorted,sorted);

  //checkCudaErrors(cudaEventRecord(seq_stop_event, 0));
  //checkCudaErrors(cudaEventSynchronize(seq_stop_event));

  //float seq_time = 0;
  //checkCudaErrors(cudaEventElapsedTime(&seq_time, seq_start_event, seq_stop_event));
  //seq_time /= 1.0e3f;
  //printf("radixSort (SEQ), Throughput = %.4f KElements/s, Time = %.5f s, Size = %u elements\n",
  //         1.0e-3f * numElements / seq_time, seq_time, numElements);


  // THRUST IMPLEMENTATION
  // copy onto GPU
  //thrust::device_vector<int> d_keys;
  //  
  //cudaEvent_t start_event, stop_event;
  //checkCudaErrors(cudaEventCreate(&start_event));
  //checkCudaErrors(cudaEventCreate(&stop_event));

  //float totalTime = 0;
  //// run multiple iterations to compute an average sort time
  //for (int i = 0; i < numIterations; i++) {
  //      // reset data before sort
  //      d_keys= h_keys;

  //      checkCudaErrors(cudaEventRecord(start_event, 0));

  //      thrust::sort(d_keys.begin(), d_keys.end());

  //      checkCudaErrors(cudaEventRecord(stop_event, 0));
  //      checkCudaErrors(cudaEventSynchronize(stop_event));

  //      float time = 0;
  //      checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
  //      totalTime += time;
  //  }

  //  totalTime /= (1.0e3f * numIterations);
  //  printf("radixSort in THRUST, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\n",
  //         1.0e-6f * numElements / totalTime, totalTime, numElements);

  //  getLastCudaError("after radixsort");

  //  // Get results back to host for correctness checking
  //  thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());

  //  getLastCudaError("copying results to host memory");

  //  // Check results
  //  bool bTestResult = thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

  //  checkCudaErrors(cudaEventDestroy(start_event));
  //  checkCudaErrors(cudaEventDestroy(stop_event));

  //  if (bTestResult) printf("THRUST: VALID!\n");

    // COMPARE SEQUENTIAL WITH THRUST
   //bTestResult = true;
   //for (int i = 0; i < (int)numElements; i++) {
   //  if (h_keysSorted[i] != sorted[i]) {
   //    bTestResult = false;
   //    break;
   //  }
   //}
   //if (bTestResult) printf("SEQ: VALID!\n");

   // TODO: NOW ADD YOUR OWN CODE, TIME AND VALIDATE AGAINST SEQUENTIAL


  //unsigned int *my_sorted;
  //my_sorted = (unsigned int *) malloc (numElements*sizeof(unsigned int));
  host_radixsort((unsigned int*)unsorted,(unsigned int*)sorted);

}

