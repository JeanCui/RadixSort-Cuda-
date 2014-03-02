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
//#define numElements 524288
#define numIterations 10

#define BLOCKSIZE 512
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
    prefix[i] = prefix[i-1] + count[i]; //wrong, should be count[i-1]
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

__global__ void global_radixsort(unsigned int *d_keys, unsigned int
    *keys_blocksorted, int *localPrefix,
    int pos, int *blockCount, int numBlocks)
{
  int tx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tx;
  // may be no need to use shared memory
  __shared__ unsigned int tile[BLOCKSIZE];
  tile[tx] = d_keys[idx];

  __shared__ int perBlockCount[BINNUM][BLOCKSIZE];

  for(int i=0;i<BINNUM;++i)
    perBlockCount[i][tx] = 0;
  
  __syncthreads();
  
  int binIdx = getBinIndex(d_keys[idx], pos);
  perBlockCount[binIdx][tx] = 1;
  __syncthreads();

  
  for(int i = BLOCKSIZE/2; i > 0; i >>= 1)
  {
    if(tx < i)
    {
      for(int j=0;j<BINNUM;++j)
      {
        perBlockCount[j][tx] = perBlockCount[j][tx]+ perBlockCount[j][i+tx];
      }
    }
     __syncthreads();
  }

  __shared__ int localPrefixSum[BINNUM];
  if(tx == 0)
  {
    for(int j=0;j<BINNUM;++j)
    {
      blockCount[blockIdx.x+j*numBlocks] = perBlockCount[j][0];
    }
    localPrefixSum[0] = 0;
    for(int j=1;j<BINNUM;++j)
    {
      localPrefixSum[j] =
        localPrefixSum[j-1]+blockCount[blockIdx.x+(j-1)*numBlocks];
    }

  }
  __syncthreads();
  if(tx < BINNUM)
  {
    localPrefix[tx+blockIdx.x*BINNUM] = localPrefixSum[tx];
    int binStartIdx = localPrefixSum[tx];
    for(int i=0;i<BLOCKSIZE;++i)
    {
      int bin = getBinIndex(tile[i],pos);
      if(tx==bin)
      {
        keys_blocksorted[binStartIdx+blockIdx.x*blockDim.x] = tile[i];
        binStartIdx++;
      }
    }

  }
  //__syncthreads();
  //if(tx==0)
  //{
  //  for(int i=0;i<BINNUM;++i)
  //    printf("block:%d bin:%d---%d\n",blockIdx.x, i,
  //      localPrefix[i+blockIdx.x*BINNUM]);
  //}

}

__global__ void combineBlockCount(int *s_data, int *g_data)
{

  int tx = threadIdx.x;
  int idx = 4*(blockIdx.x * blockDim.x + tx);
  extern __shared__ int data[];
  data[4*tx] = s_data[idx];
  data[4*tx+1] = s_data[idx+1];
  data[4*tx+2] = s_data[idx+2];
  data[4*tx+3] = s_data[idx+3];

  data[4*tx] = data[4*tx]+data[4*tx+1]+data[4*tx+2]+data[4*tx+3];
 // printf("block:%d, bin:%d--%d\n",tx, blockIdx.x, data[tx]);
  __syncthreads();

  for(int i=blockDim.x/2; i > 0; i >>= 1)
  {
    if(tx < i)
      data[4*tx] += data[4*(i+tx)];

    __syncthreads();
  }

  if(tx == 0)
  {
    g_data[blockIdx.x] = data[0];
  }
}

__global__ void combineBlockCount2(int *s_data, int *g_data)
{

  int tx = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + tx);
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

__global__ void prescan(int *s_data, int *g_data)
{
  extern __shared__ int temp[];
  int tx = threadIdx.x;
  int offset = 1;

  temp[2*tx] = s_data[2*tx];
  temp[2*tx+1] = s_data[2*tx+1];

  for(int d = BINNUM>>1; d > 0;d>>=1)
  {
    __syncthreads();
    if(tx<d)
    {
      int ai = offset*(2*tx+1)-1;
      int bi = offset*(2*tx+2)-1;

      temp[bi] += temp[ai];
    }
    offset*=2;
  }

  if(tx==0)
    temp[BINNUM-1] = 0;

  for(int d=1;d<BINNUM;d*=2)
  {
    offset >>= 1;
    __syncthreads();
    if(tx<d)
    {
      int ai = offset*(2*tx+1)-1;
      int bi = offset*(2*tx+2)-1;

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  
  g_data[2*tx] = temp[2*tx];
  g_data[2*tx+1] = temp[2*tx+1];
  
}
__global__ void scatter(unsigned int *s_data, unsigned int *g_data, int
    *blockCount, int *localPrefix, int *globalPrefix, int pos, int numBlocks)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = bx*blockDim.x+tx;

  unsigned int element = s_data[idx];
  int bin = getBinIndex(element,pos);
  
  // blockCount[BINNUM][NUMBLOCKS]
  // localPrefix[NUMBLOCKS][BINNUM]
  // globalPrefix[BINNUM]

  int di; // final offset in output array
  di = tx - localPrefix[blockIdx.x*BINNUM+bin] + globalPrefix[bin];

  int ci=0;
  for(int i=0;i<bx;++i)
    ci += blockCount[bin*numBlocks+i];
  di += ci;

  g_data[di] = element;

}
__host__ void host_radixsort(unsigned int *h_keys, unsigned int *h_sorted)
{
  unsigned int *d_keys;
  unsigned int numbytes = numElements*sizeof(unsigned int);

  checkCudaErrors(cudaMalloc((void **) &d_keys, numbytes));
  checkCudaErrors(cudaMemcpy(d_keys, h_keys, numbytes, cudaMemcpyHostToDevice));
 // checkCudaErrors(cudaMalloc((void**)&d_keysSorted, numbytes));
 // checkCudaErrors(cudaMemset(d_keysSorted, 0, numbytes));


  int numBlocks = numElements / BLOCKSIZE;
 
  cudaEvent_t my_start_event, my_stop_event;
  checkCudaErrors(cudaEventCreate(&my_start_event));
  checkCudaErrors(cudaEventCreate(&my_stop_event));
 
  checkCudaErrors(cudaEventRecord(my_start_event, 0));
  int *blockCount;
  checkCudaErrors(cudaMalloc((void**)&blockCount, BINNUM*numBlocks*sizeof(int)));
  int *overallCount;
  checkCudaErrors(cudaMalloc((void**)&overallCount, BINNUM*sizeof(int)));
  
  int *localPrefixSum;
  checkCudaErrors(cudaMalloc((void**)&localPrefixSum, BINNUM*numBlocks*sizeof(int)));

  int *prefixSumArray;
  checkCudaErrors(cudaMalloc((void**)&prefixSumArray, BINNUM*sizeof(int)));

  unsigned int *keys_blocksorted;
  checkCudaErrors(cudaMalloc((void**)&keys_blocksorted, numbytes));

  unsigned int *keys_scatter;
  checkCudaErrors(cudaMalloc((void**)&keys_scatter, numbytes));
  
  for(int i = 0; i < 32; i+=4)
  {
    global_radixsort<<<numBlocks, BLOCKSIZE>>>(d_keys, keys_blocksorted, localPrefixSum,
        i, blockCount, numBlocks);
  
    combineBlockCount<<<BINNUM, numBlocks/4,
        numBlocks*sizeof(int)>>>(blockCount, overallCount);
    //combineBlockCount2<<<BINNUM, numBlocks,
    //    numBlocks*sizeof(int)>>>(blockCount, overallCount);
    // prefix sum
    prescan<<<1,BINNUM/2, BINNUM*sizeof(int)>>>(overallCount, prefixSumArray);
    
    //scatter to d_keys
    scatter<<<numBlocks, BLOCKSIZE>>>(keys_blocksorted, keys_scatter, blockCount,
        localPrefixSum, prefixSumArray,i, numBlocks);

    checkCudaErrors(cudaMemcpy(d_keys, keys_scatter, numbytes,
          cudaMemcpyDeviceToDevice));

  }

  checkCudaErrors(cudaEventRecord(my_stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(my_stop_event));
  float my_time = 0;
  checkCudaErrors(cudaEventElapsedTime(&my_time, my_start_event, my_stop_event));
  my_time /= 1.0e3f;
  printf("radixSort (MyTest), Throughput = %.4f KElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-3f * numElements / my_time, my_time, numElements);

//#define testreorder

#ifdef testblockcount
  int *hostBlockCount;
  hostBlockCount = (int*)malloc(BINNUM*numBlocks*sizeof(int));
  checkCudaErrors(cudaMemcpy(hostBlockCount, blockCount,
        BINNUM*numBlocks*sizeof(int), cudaMemcpyDeviceToHost));
  for(int i = 0;i<numBlocks;++i)
  {
    for(int j=0;j<BINNUM;++j)
      printf("block:%d bin%d---%d\n",i, j, hostBlockCount[i+j*numBlocks]);
  }
#endif

#ifdef testlocalprefix
  int *hostLocalPrefix;
  hostLocalPrefix = (int*)malloc(BINNUM*numBlocks*sizeof(int));
  checkCudaErrors(cudaMemcpy(hostLocalPrefix, localPrefixSum,
        BINNUM*numBlocks*sizeof(int), cudaMemcpyDeviceToHost));

  for(int i=0;i<numBlocks;++i)
    for(int j=0;j<BINNUM;++j)
      printf("block:%d prefix:%d---%d\n",i, j, hostLocalPrefix[j+i*BINNUM]);

#endif
#ifdef testsortedblock
  unsigned int *hostBlockSorted;
  hostBlockSorted = (unsigned int*)malloc(numbytes);
  checkCudaErrors(cudaMemcpy(hostBlockSorted, keys_blocksorted,
        numbytes, cudaMemcpyDeviceToHost));
  
  for(int i=0; i < numElements;++i)
    printf("%d\n",hostBlockSorted[i]);
  
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

#ifdef testscatter
  unsigned int *hostScatter;
  hostScatter = (unsigned int*)malloc(numbytes);
  checkCudaErrors(cudaMemcpy(hostScatter, keys_scatter,
        numbytes, cudaMemcpyDeviceToHost));
  
  for(int i=0; i < numElements;++i)
    printf("%d\n",hostScatter[i]);
#endif

#ifdef testreorder
  unsigned int *hostKeys;
  hostKeys = (unsigned int*)malloc(numbytes);
  checkCudaErrors(cudaMemcpy(hostKeys, d_keys,
        numbytes, cudaMemcpyDeviceToHost));
  
  for(int i=0; i < numElements;++i)
    printf("%d\n",hostKeys[i]);
#endif



  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_sorted, d_keys, numbytes, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_keys));              
  checkCudaErrors(cudaFree(blockCount));              
  checkCudaErrors(cudaFree(overallCount));              
  checkCudaErrors(cudaFree(localPrefixSum));              
  checkCudaErrors(cudaFree(prefixSumArray));              
  checkCudaErrors(cudaFree(keys_blocksorted));              
  checkCudaErrors(cudaFree(keys_scatter));              

  //checkCudaErrors(cudaFree(d_keysSorted));
  
}

int
main(int argc, char **argv)
{
  int *unsorted;//, *sorted;

  // initialize list.  Value in range 0..RADIX
  unsorted = (int *) malloc (numElements*sizeof(int));
  //sorted = (int *) malloc (numElements*sizeof(int));
  for (int i=0; i<numElements; i++) {
    
    unsorted[i] = (int) (rand() % RADIX);
   // if(i%BLOCKSIZE < (BLOCKSIZE/2))
   //   unsorted[i] = 2;
   // else
   //   unsorted[i] = 1;
  }

  //initialize list for Thrust
  thrust::host_vector<int> h_keys(numElements);
  thrust::host_vector<int> h_keysSorted(numElements);
  for (int i = 0; i < (int)numElements; i++)
     h_keys[i] = unsorted[i];

  //// SEQUENTIAL RUN
  //cudaEvent_t seq_start_event, seq_stop_event;
  //checkCudaErrors(cudaEventCreate(&seq_start_event));
  //checkCudaErrors(cudaEventCreate(&seq_stop_event));
  //checkCudaErrors(cudaEventRecord(seq_start_event, 0));

  //// TODO: THIS TAKES A FEW MINUTES AND SHOULD BE COMMENTED OUT FOR TESTING
  //(void) sequentialSort(unsorted,sorted);

  //checkCudaErrors(cudaEventRecord(seq_stop_event, 0));
  //checkCudaErrors(cudaEventSynchronize(seq_stop_event));

  //float seq_time = 0;
  //checkCudaErrors(cudaEventElapsedTime(&seq_time, seq_start_event, seq_stop_event));
  //seq_time /= 1.0e3f;
  //printf("radixSort (SEQ), Throughput = %.4f KElements/s, Time = %.5f s, Size = %u elements\n",
  //         1.0e-3f * numElements / seq_time, seq_time, numElements);


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
   //bTestResult = true;
   //for (int i = 0; i < (int)numElements; i++) {
   //  if (h_keysSorted[i] != sorted[i]) {
   //    bTestResult = false;
   //    break;
   //  }
   //}
   //if (bTestResult) printf("SEQ: VALID!\n");

   // TODO: NOW ADD YOUR OWN CODE, TIME AND VALIDATE AGAINST SEQUENTIAL


  unsigned int *my_sorted;
  my_sorted = (unsigned int *) malloc (numElements*sizeof(unsigned int));
  host_radixsort((unsigned int*)unsorted, my_sorted);
  bool MyTestResult = true;
  for (int i = 0; i < (int)numElements; i++) {
    if (my_sorted[i] != h_keysSorted[i]) {
      MyTestResult = false;
      break;
    }
  }
  if (MyTestResult) printf("MyVersion: VALID!\n");

}

