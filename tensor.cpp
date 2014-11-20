#include <TH.h>
#include <luaT.h>
#include <THC/THC.h>

#include <stdexcept>
#include <map>
#include <string>

#include "metrics.h"


inline void luaAssert (bool condition, const char *message) {
 
  if(!condition)
    throw std::invalid_argument(message);  
}



#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor        TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define libtensor_(NAME) TH_CONCAT_4(libtensor_, Real, _, NAME)

#include "generic/tensor.cpp"
#include "THGenerateAllTypes.h"

#include "generic/tensor.cpp"
#include "THGenerateCudaTypes.h"



//============================================================
// Register functions in LUA
//
static const luaL_reg libtensor_ext_init [] =
{  
  {NULL,NULL}
};


extern "C" {

  DLL_EXPORT int luaopen_libtensor_ext(lua_State *L)
  {
    libtensor_Char_init(L);
    libtensor_Byte_init(L);
    libtensor_Short_init(L);
    libtensor_Int_init(L);
    libtensor_Long_init(L);
    libtensor_Float_init(L);
    libtensor_Double_init(L);
    libtensor_Cuda_init(L);

    luaL_register(L, "libtensor_ext", libtensor_ext_init);    
    return 1;
  }

}