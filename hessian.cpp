extern "C" {
#include<luaT.h>
#include<TH/TH.h>
}
#include <cassert>
#include <gsl/gsl_block.h>
#include <gsl/gsl_linalg.h>
using namespace std;

template<typename T> inline T min(T a, T b) {
  return (a < b) ? a : b;
}

typedef THFloatTensor Tensor;
#define ID_TENSOR_STRING "torch.FloatTensor"
#define Tensor_(a) THFloatTensor_##a
typedef float Real;
typedef double accreal;

static int QR(lua_State *L) {
  const char* iddouble = "torch.DoubleTensor";
  THDoubleTensor* input   = (THDoubleTensor*)luaT_checkudata(L, 1, iddouble);
  THDoubleTensor* outputQ = (THDoubleTensor*)luaT_checkudata(L, 2, iddouble);
  THDoubleTensor* outputR = (THDoubleTensor*)luaT_checkudata(L, 3, iddouble);

  long    h   = input->size[0];
  long    w   = input->size[1];
  long*   is  = input  ->stride;
  long*   oqs = outputQ->stride;
  long*   ors = outputR->stride;
  double* ip  = THDoubleTensor_data(input);
  double* oqp = THDoubleTensor_data(outputQ);
  double* orp = THDoubleTensor_data(outputR);
  
  gsl_matrix ig;
  ig.size1 = h;
  ig.size2 = w;
  ig.tda   = is[0];
  assert(is[1] == 1);
  ig.data  = ip;
  ig.block = NULL;
  ig.owner = 0;
  gsl_matrix oqg;
  oqg.size1 = h;
  oqg.size2 = w;
  oqg.tda   = oqs[0];
  assert(oqs[1] == 1);
  oqg.data  = oqp;
  oqg.block = NULL;
  oqg.owner = 0;
  gsl_matrix org;
  org.size1 = h;
  org.size2 = w;
  org.tda   = ors[0];
  assert(ors[1] == 1);
  org.data  = orp;
  org.block = NULL;
  org.owner = 0;
  gsl_vector tau;
  tau.size = min(h, w);
  tau.stride = 1;
  tau.data = new double[tau.size];
  tau.block = NULL;
  tau.owner = 0;
  
  gsl_linalg_QR_decomp(&ig, &tau);
  gsl_linalg_QR_unpack(&ig, &tau, &oqg, &org);

  delete[] tau.data;
  
  return 0;
}

static const struct luaL_reg libhessian[] = {
  {"QR", QR},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libhessian(lua_State *L) {
  luaL_openlib(L, "libhessian", libhessian, 0);
  return 1;
}
