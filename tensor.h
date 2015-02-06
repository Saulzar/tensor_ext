#ifndef TENSOR_H
#define TENSOR_H


template<typename T>
struct mod_functor {
  const T p;

  mod_functor(T p_) : p(p_) {}
    __host__ __device__ T operator()(const T& x) const
  {
    return x % p;
  }  
};


template<>
struct mod_functor<float>
{
  const float p;

  mod_functor(float p_) : p(p_) {}
    __host__ __device__ float operator()(const float& x) const
  {
    return fmod(x, p);
  }
};

template<>
struct mod_functor<double>
{
  const double p;

  mod_functor(double p_) : p(p_) {}
    __host__ __device__ double operator()(const double& x) const
  {
    return fmodl(x, p);
  }
};



template<typename T>
struct clamp_functor
{
  const T lower, upper;

  clamp_functor(T lower_, T upper_) : lower(lower_), upper(upper_) {}
    __host__ __device__ T operator()(const T& x) const
  {
    return (x < lower) ? lower : (x > upper ? upper : x);
  }
};


template<typename T>
struct min_functor
{
  const T lower;

  min_functor(T lower_) : lower(lower_) {}
    __host__ __device__ T operator()(const T& x) const
  {
    return (x < lower) ? lower : x;
  }
};

template<typename T>
struct max_functor
{
  const T upper;

  max_functor(T upper_) : upper(upper_) {}
    __host__ __device__ T operator()(const T& x) const
  {
    return x > upper ? upper : x;
  }
};




void libtensor_Cuda_max(THCState *state, THCudaTensor *self_, THCudaTensor *src_, float upper);
void libtensor_Cuda_min(THCState *state, THCudaTensor *self_, THCudaTensor *src_, float lower);


void libtensor_Cuda_indexSum(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src);
void libtensor_Cuda_clamp(THCState *state, THCudaTensor *self_, THCudaTensor *src_, float lower, float upper);

void libtensor_Cuda_mod(THCState *state, THCudaTensor *self_, THCudaTensor *src_, float p);
  
  
  
#endif
