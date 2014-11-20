
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/tensor.cpp"
#else


#include <luaT.h>
#include "tensor.h"

#ifndef TH_REAL_IS_CUDA

#include <TH.h>
#include <TH/generic/THTensorMath.h>


inline THTensor *libtensor_(checkTensor)(lua_State* L, int arg) {
  return (THTensor*)luaT_checkudata(L, arg, torch_Tensor);  
}

void libtensor_(indexSum)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  long i, numel;
  THTensor *tSlice, *sSlice;
  long *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(numel == src->size[dim],4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->nDimension > 1 ) {
    for (i=0; i<numel; i++) {
      tSlice = THTensor_(new)();
      sSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor, dim, index_data[i]-1);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(cadd)(tSlice, tSlice, 1, sSlice);
      THTensor_(free)(tSlice);
      THTensor_(free)(sSlice);
    }
  } else {
    
    for (i=0; i<numel; i++) {
      long idx = index_data[i]-1;
      THTensor_(set1d)(tensor, idx, THTensor_(get1d)(src,i) + THTensor_(get1d)(tensor, idx));
    }
  }
  
  THLongTensor_free(index);
}


template<typename Op>
void libtensor_(transform)(THTensor *r_, THTensor *t, Op const &op)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      real t_val;
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++) {
	rp[i] = op(tp[i]);
      }
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = op(*t_data););
  }
}





void libtensor_(clamp)(THTensor *r_, THTensor *t, real min_value, real max_value) {
  libtensor_(transform)(r_, t, clamp_functor<real>(min_value, max_value));
}

void libtensor_(mod)(THTensor *r_, THTensor *t, real p) {
  libtensor_(transform)(r_, t, mod_functor<real>(p));
}

#endif


static int libtensor_(luaMod)(lua_State *L) {
  
  int narg = lua_gettop(L);
  if(!(narg == 2 || narg == 3)) {
    luaL_error(L, "mod: expected 2 or 3 arguments"); 
  }
  
  THTensor *r_ =   (THTensor*)luaT_checkudata(L, 1, torch_Tensor);
  THTensor *t =   (THTensor*)luaT_checkudata(L, narg - 1, torch_Tensor);

  real p = lua_tonumber(L, narg);
  THArgCheck(p > 0, narg, "mod: divisor must be > 0");
  
  libtensor_(mod)(r_, t, p);
  
  THTensor_(retain)(r_);
  luaT_pushudata(L, r_, torch_Tensor);
  
  return 1;
}


static int libtensor_(luaClamp)(lua_State *L) {
  
  int narg = lua_gettop(L);
  
  if(!(narg == 3 || narg == 4)) {
    luaL_error(L, "clamp: expected 3 or 4 arguments"); 
  }
  
  THTensor *r_ =   (THTensor*)luaT_checkudata(L, 1, torch_Tensor);
  THTensor *t =   (THTensor*)luaT_checkudata(L, narg - 2, torch_Tensor);

  real min_value = lua_tonumber(L, narg - 1);
  real max_value = lua_tonumber(L, narg);       


  libtensor_(clamp)(r_, t, min_value, max_value);
  
  THTensor_(retain)(r_);
  luaT_pushudata(L, r_, torch_Tensor);
  
  return 1;
}

static int libtensor_(luaIndexSum)(lua_State* L) {
     
  int args = lua_gettop(L);
  if(args < 4) {
    luaL_error(L, "indexSum: expected 4 arguments"); 
  }
  
  THTensor *r_ =   (THTensor*)luaT_checkudata(L, 1, torch_Tensor);
  int dim = lua_tonumber(L, 2) - 1;     
  THLongTensor *index =   (THLongTensor*)luaT_checkudata(L, 3, "torch.LongTensor");  
  THTensor *src =   (THTensor*)luaT_checkudata(L, 4, torch_Tensor);  
  
    
  libtensor_(indexSum)(r_, dim, index, src);  
  
  THTensor_(retain)(r_);
  luaT_pushudata(L, r_, torch_Tensor); 
  
  
  return 1;
}

//============================================================
// Register functions in LUA
//
static const luaL_reg libtensor_(Main__) [] =
{
  {"indexSum", libtensor_(luaIndexSum)},
  {"clamp", libtensor_(luaClamp)},
  {"mod", libtensor_(luaMod)},
  {NULL, NULL}  /* sentinel */
};


extern "C" {

  DLL_EXPORT int libtensor_(init) (lua_State *L) {
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, libtensor_(Main__), "libtensor_ext");
    lua_pop(L,1); 
    return 1;
  }

}

#endif

