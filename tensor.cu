// Includes

#include <luaT.h>
#include <TH.h>

#include <THC/THC.h>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>

#include "tensor.h"

#include <sstream>


#define cudaAssert(ans) { cudaAssert_((ans), __FILE__, __LINE__); }
inline void cudaAssert_(cudaError_t code, const char *file, int line)
{

  if (code != cudaSuccess) 
  {
    std::ostringstream out;
    out << "cuda error " << file << ":" << line << " " << cudaGetErrorString(code);
    
    throw std::logic_error(out.str());
    
  }
}

template<typename Op>
__global__ void kernel_indexReduce(
   float *res, float *src, long* res_stride, float *index,
   long res_nDim, int dim, long idx_size, long src_size, long size_dim, Op const &op
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = src_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int resIdx = 0;
      for (int d=0; d<res_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = res_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          resIdx += coeff * res_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / res_stride[d];
          leftover -= coeff * res_stride[d];
          targetIdx += coeff * res_stride[d];
          resIdx += coeff * res_stride[d];
        }
      }
      
      int r = resIdx + ((int)(index[i])-1)*res_stride[dim];
      op(res[r], src[targetIdx + i*res_stride[dim]]);
    }
  }
}

template<typename Op>
void indexReduce(THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src, Op const &op)
{
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nSrc;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THCudaTensor_newContiguous(src);
  indices_ = THCudaTensor_newWithSize1d(nIndex);
  THCudaTensor_copyLong(indices_, indices);

  nSrc = THCudaTensor_nElement(src);
  
  const int size = 16;
  dim3 nthreads(size, size);
  dim3 nblocks(ceil((float)(nSrc / nIndex) / (size*size)));
  

  THCudaCheck(cudaMalloc((void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  kernel_indexReduce<<<nblocks, nthreads>>>(
    THCudaTensor_data(res_), THCudaTensor_data(src),
    stride_, THCudaTensor_data(indices_),
    res_->nDimension, dim, nIndex,
    THCudaTensor_nElement(src), res_->size[dim], op
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(indices_);
  THCudaTensor_free(src);
}


template<typename Op>
void transform(THCudaTensor *self_, THCudaTensor *src_, Op const &op)
{
  THCudaTensor_resizeAs(self_, src_);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  THCudaTensor *src = THCudaTensor_newContiguous(src_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  thrust::transform(src_data, src_data+size, self_data, op);

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}



struct add_functor {
   
  __device__ void operator()(float& x, float& y) const {
    x += y;
  }  
};



void libtensor_Cuda_clamp(THCudaTensor *self_, THCudaTensor *src_, float lower, float upper) {
  transform(self_, src_, clamp_functor<float>(lower, upper));
}

void libtensor_Cuda_mod(THCudaTensor *self_, THCudaTensor *src_, float p) {
  transform(self_, src_, mod_functor<float>(p));
}


void libtensor_Cuda_indexSum(THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src) {
  indexReduce(res_, dim, indices, src, add_functor());
}


