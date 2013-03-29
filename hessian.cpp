extern "C" {
#include<luaT.h>
#include<TH/TH.h>
}
#include<cmath>
#include <cassert>
#include<vector>
#include<algorithm>
#include <gsl/gsl_block.h>
#include <gsl/gsl_linalg.h>
using namespace std;

template<typename T> inline T sq(T a) {
  return a*a;
}

typedef THDoubleTensor Tensor;
#define ID_TENSOR_STRING "torch.DoubleTensor"
#define Tensor_(a) THDoubleTensor_##a
typedef double Real;
typedef double accreal;

static int Spaghetti_updateOutput(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor* input   = (Tensor*)luaT_checkudata(L, 1, idreal);
  THLongTensor* conSrc  = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* conDst  = (THLongTensor*)luaT_checkudata(L, 3, idlong);
  Tensor* weights = (Tensor*)luaT_checkudata(L, 4, idreal);
  Tensor* output  = (Tensor*)luaT_checkudata(L, 5, idreal);

  long nCon  = conSrc->size[0];
  long nDims = conSrc->size[1];
  long* is  = input  ->stride;
  long* css = conSrc ->stride;
  long* cds = conDst ->stride;
  long* ws  = weights->stride;
  long* os  = output ->stride;
  Real* ip  = Tensor_(data)(input  );
  long* csp = THLongTensor_data(conSrc );
  long* cdp = THLongTensor_data(conDst );
  Real* wp  = Tensor_(data)(weights);
  Real* op  = Tensor_(data)(output );

  Tensor_(zero)(output);

  int i, j;
  long sidx, didx;
#pragma omp parallel for private(i, j, sidx, didx)
  for (i = 0; i < nCon; ++i) {
    sidx = didx = 0;
    for (j = 0; j < nDims; ++j) {
      sidx += is[j] * (csp[i*css[0] + j*css[1]] - 1);
      didx += os[j] * (cdp[i*cds[0] + j*cds[1]] - 1);
    }
    op[didx] += wp[ws[0]*i] * ip[sidx];
  }
  
  return 0;
}

static int Spaghetti_updateGradInput(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor* input      = (Tensor*)luaT_checkudata(L, 1, idreal);
  THLongTensor* conSrc     = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* conDst     = (THLongTensor*)luaT_checkudata(L, 3, idlong);
  Tensor* weights    = (Tensor*)luaT_checkudata(L, 4, idreal);
  Tensor* gradOutput = (Tensor*)luaT_checkudata(L, 5, idreal);
  Tensor* gradInput  = (Tensor*)luaT_checkudata(L, 6, idreal);

  long nCon  = conSrc->size[0];
  long nDims = conSrc->size[1];
  long* is  = input     ->stride;
  long* css = conSrc    ->stride;
  long* cds = conDst    ->stride;
  long* ws  = weights   ->stride;
  long* gos = gradOutput->stride;
  long* gis = gradInput ->stride;
  Real* ip  = Tensor_(data)(input     );
  long* csp = THLongTensor_data(conSrc    );
  long* cdp = THLongTensor_data(conDst    );
  Real* wp  = Tensor_(data)(weights   );
  Real* gop = Tensor_(data)(gradOutput);
  Real* gip = Tensor_(data)(gradInput );

  Tensor_(zero)(gradInput);

  int i, j;
  long sidx, didx;
  for (i = 0; i < nCon; ++i) {
    sidx = didx = 0;
    for (j = 0; j < nDims; ++j) {
      sidx += gis[j] * (csp[i*css[0] + j*css[1]] - 1);
      didx += gos[j] * (cdp[i*cds[0] + j*cds[1]] - 1);
    }
    gip[sidx] += wp[ws[0]*i] * gop[didx];
  }
  
  return 0;
}

static int Spaghetti_accGradParameters(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor* input      = (Tensor*)luaT_checkudata(L, 1, idreal);
  THLongTensor* conSrc     = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* conDst     = (THLongTensor*)luaT_checkudata(L, 3, idlong);
  Tensor* weights    = (Tensor*)luaT_checkudata(L, 4, idreal);
  Tensor* gradOutput = (Tensor*)luaT_checkudata(L, 5, idreal);
  Real    scale      =          lua_tonumber   (L, 6);
  Tensor* gradWeight = (Tensor*)luaT_checkudata(L, 7, idreal);

  long nCon  = conSrc->size[0];
  long nDims = conSrc->size[1];
  long* is  = input     ->stride;
  long* css = conSrc    ->stride;
  long* cds = conDst    ->stride;
  long* ws  = weights   ->stride;
  long* gos = gradOutput->stride;
  long* gws = gradWeight->stride;
  Real* ip  = Tensor_(data)(input     );
  long* csp = THLongTensor_data(conSrc    );
  long* cdp = THLongTensor_data(conDst    );
  Real* wp  = Tensor_(data)(weights   );
  Real* gop = Tensor_(data)(gradOutput);
  Real* gwp = Tensor_(data)(gradWeight);

  int i, j;
  long sidx, didx;
#pragma omp parallel for private(i, sidx, didx)
  for (i = 0; i < nCon; ++i) {
    sidx = didx = 0;
    for (j = 0; j < nDims; ++j) {
      didx += gos[j] * (cdp[i*cds[0] + j*cds[1]] - 1);
      sidx += is [j] * (csp[i*css[0] + j*css[1]] - 1);
    }
    gwp[i*gws[0]] += scale * ip[sidx] * gop[didx];
  }
  
  return 0;
}

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

static int BfNormalize(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  Tensor* weights      = (Tensor*)luaT_checkudata(L, 1, idreal);
  
  int n = weights->size[0];
  long* ws = weights->stride;
  Real* wp = Tensor_(data)(weights);

  int i;
  Real c, s, normalizer;
  for (i = 0; i < n; i += 4) {
    c = 0.5 * (wp[ws[0]*i  ] + wp[ws[0]*i+3]);
    s = 0.5 * (wp[ws[0]*i+1] - wp[ws[0]*i+2]);
    normalizer = 1. / sqrt(c*c+s*s);
    wp[ws[0]*i  ] =  c*normalizer;
    wp[ws[0]*i+1] =  s*normalizer;
    wp[ws[0]*i+2] = -s*normalizer;
    wp[ws[0]*i+3] =  c*normalizer;
  }
  
  return 0;
}

static int BfHalfNormalize(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  Tensor* weights      = (Tensor*)luaT_checkudata(L, 1, idreal);
  
  int n = weights->size[0];
  long* ws = weights->stride;
  Real* wp = Tensor_(data)(weights);

  int i;
  Real c, s;
  for (i = 0; i < n; i += 4) {
    c = 0.5 * (wp[ws[0]*i  ] + wp[ws[0]*i+3]);
    s = 0.5 * (wp[ws[0]*i+1] - wp[ws[0]*i+2]);
    wp[ws[0]*i  ] =  c;
    wp[ws[0]*i+1] =  s;
    wp[ws[0]*i+2] = -s;
    wp[ws[0]*i+3] =  c;
  }
  
  return 0;
}

static int BfDistanceToNormalizeAccGrad(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  Tensor* weights      = (Tensor*)luaT_checkudata(L, 1, idreal);
  Tensor* gradWeights  = (Tensor*)luaT_checkudata(L, 2, idreal);
  Real    lambda       =          lua_tonumber(L, 3);
  Real    ntot         =          lua_tonumber(L, 4);
  
  int n = weights->size[0];
  long*  ws = weights->stride;
  long* gws = gradWeights->stride;
  Real*  wp = Tensor_(data)(weights);
  Real* gwp = Tensor_(data)(gradWeights);

  int i;
  Real a, b, d, d0 = - 4.*lambda / ntot;
  for (i = 0; i < n; i += 2) {
    a = wp[ws[0]*i];
    b = wp[ws[0]*(i+1)];
    d = d0 * (1. - sq(a) - sq(b));
    gwp[gws[0]*i]     += d*a;
    gwp[gws[0]*(i+1)] += d*b;
  }
  
  return 0;
}

static const struct luaL_reg libhessian[] = {
  {"QR", QR},
  {"spaghetti_updateOutput", Spaghetti_updateOutput},
  {"spaghetti_updateGradInput", Spaghetti_updateGradInput},
  {"spaghetti_accGradParameters", Spaghetti_accGradParameters},
  {"bfNormalize", BfNormalize},
  {"bfHalfNormalize", BfHalfNormalize},
  {"bfDistanceToNormalizeAccGrad", BfDistanceToNormalizeAccGrad},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libhessian(lua_State *L) {
  luaL_openlib(L, "libhessian", libhessian, 0);
  return 1;
}
