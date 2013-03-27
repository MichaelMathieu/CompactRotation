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
  Tensor*       input   = (Tensor*      )luaT_checkudata(L, 1, idreal);
  THLongTensor* conSrc  = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* conDst  = (THLongTensor*)luaT_checkudata(L, 3, idlong);
  THLongTensor* conWei  = (THLongTensor*)luaT_checkudata(L, 4, idlong);
  Tensor*       weights = (Tensor*      )luaT_checkudata(L, 5, idreal);
  Tensor*       output  = (Tensor*      )luaT_checkudata(L, 6, idreal);
  THLongTensor* order   = (THLongTensor*)luaT_checkudata(L, 7, idlong);
  THLongTensor* orderChk= (THLongTensor*)luaT_checkudata(L, 8, idlong);

  const long nChunks = orderChk->size[0]-1;
  const long nDims = conSrc->size[1];
  const long* is  = input  ->stride;
  const long  css0= conSrc ->stride[0];
  const long  cds0= conDst ->stride[0];
  const long  ws  = weights->stride[0];
  const long* os  = output ->stride;
  const Real* ip  = Tensor_(data)(input);
  const long* csp = THLongTensor_data(conSrc);
  const long* cdp = THLongTensor_data(conDst);
  const long* cwp = THLongTensor_data(conWei);
  const Real* wp  = Tensor_(data)(weights);
  Real* op  = Tensor_(data)(output);
  const long* orp  = THLongTensor_data(order);
  const long* orcp = THLongTensor_data(orderChk);

  Tensor_(zero)(output);

  /*
  int i, j, iChk;
  long sidx, didx;
#pragma omp parallel for private(iChk, i, j, sidx, didx)
  for (iChk = 0; iChk < nChunks; ++iChk) {
    const int idst = orcp[iChk+1];
    for (i = orcp[iChk]; i < idst; ++i) {
      const int k = orp[i];
      sidx = didx = 0;
      for (j = 0; j < nDims; ++j) {
	sidx += is[j] * csp[k*css0 + j];
	didx += os[j] * cdp[k*cds0 + j];
      }
      op[didx] += wp[ws*cwp[k]] * ip[sidx];
    }
    }
  */
  int i, j, iChk;
#pragma omp parallel for private(iChk, i, j)
  for (iChk = 0; iChk < nChunks; ++iChk) {
    const int idst = orcp[iChk+1];
    for (i = orcp[iChk]; i < idst; ++i) {
      const int k = orp[i];
      op[os[0]*cdp[k*cds0+j]] += wp[ws*cwp[k]] * ip[is[0]*csp[k*css0+j]];
    }
  }
  
  return 0;
}

static int Spaghetti_updateGradInput(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor*       input      = (Tensor*      )luaT_checkudata(L, 1, idreal);
  THLongTensor* conSrc     = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* conDst     = (THLongTensor*)luaT_checkudata(L, 3, idlong);
  THLongTensor* conWei     = (THLongTensor*)luaT_checkudata(L, 4, idlong);
  Tensor*       weights    = (Tensor*      )luaT_checkudata(L, 5, idreal);
  Tensor*       gradOutput = (Tensor*      )luaT_checkudata(L, 6, idreal);
  Tensor*       gradInput  = (Tensor*      )luaT_checkudata(L, 7, idreal);
  THLongTensor* order   = (THLongTensor*)luaT_checkudata(L, 8, idlong);
  THLongTensor* orderChk= (THLongTensor*)luaT_checkudata(L, 9, idlong);

  const long nChunks = orderChk->size[0]-1;
  const long nDims = conSrc->size[1];
  const long* is  = input     ->stride;
  const long  css0= conSrc    ->stride[0];
  const long  cds0= conDst    ->stride[0];
  const long  ws  = weights   ->stride[0];
  const long* gos = gradOutput->stride;
  const long* gis = gradInput ->stride;
  const Real* ip  = Tensor_(data)(input);
  const long* csp = THLongTensor_data(conSrc);
  const long* cdp = THLongTensor_data(conDst);
  const long* cwp = THLongTensor_data(conWei);
  const Real* wp  = Tensor_(data)(weights);
  const Real* gop = Tensor_(data)(gradOutput);
  Real* gip = Tensor_(data)(gradInput);
  const long* orp  = THLongTensor_data(order);
  const long* orcp = THLongTensor_data(orderChk);

  Tensor_(zero)(gradInput);

  /*
  int i, j, iChk;
  long sidx, didx;
#pragma omp parallel for private(iChk, i, j, sidx, didx)
  for (iChk = 0; iChk < nChunks; ++iChk) {
    const int idst = orcp[iChk+1];
    for (i = orcp[iChk]; i < idst; ++i) {
      const int k = orp[i];
      sidx = didx = 0;
      for (j = 0; j < nDims; ++j) {
	sidx += gis[j] * csp[k*css0 + j];
	didx += gos[j] * cdp[k*cds0 + j];
      }
      gip[sidx] += wp[ws*cwp[k]] * gop[didx];
    }
  }
  */
  
  int i, j, iChk;
#pragma omp parallel for private(iChk, i, j)
  for (iChk = 0; iChk < nChunks; ++iChk) {
    const int idst = orcp[iChk+1];
    for (i = orcp[iChk]; i < idst; ++i) {
      const int k = orp[i];
      gip[is[0]*csp[k*css0+j]] += wp[ws*cwp[k]] * gop[gos[0]*cdp[k*cds0+j]];
    }
  }

  
  return 0;
}

static int Spaghetti_accGradParameters(lua_State* L) {
  const char* idreal = ID_TENSOR_STRING;
  const char* idlong = "torch.LongTensor";
  Tensor*       input      = (Tensor*      )luaT_checkudata(L, 1, idreal);
  THLongTensor* conSrc     = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* conDst     = (THLongTensor*)luaT_checkudata(L, 3, idlong);
  THLongTensor* conWei     = (THLongTensor*)luaT_checkudata(L, 4, idlong);
  Tensor*       gradOutput = (Tensor*      )luaT_checkudata(L, 5, idreal);
  Real          scale      =                lua_tonumber   (L, 6);
  Tensor*       gradWeight = (Tensor*      )luaT_checkudata(L, 7, idreal);
  THLongTensor* order   = (THLongTensor*)luaT_checkudata(L, 8, idlong);
  THLongTensor* orderChk= (THLongTensor*)luaT_checkudata(L, 9, idlong);

  const long nChunks = orderChk->size[0]-1;
  const long nDims = conSrc->size[1];
  const long* is  = input     ->stride;
  const long css0= conSrc    ->stride[0];
  const long cds0= conDst    ->stride[0];
  const long* gos = gradOutput->stride;
  const long gws = gradWeight->stride[0];
  const Real* ip  = Tensor_(data)(input);
  const long* csp = THLongTensor_data(conSrc);
  const long* cdp = THLongTensor_data(conDst);
  const long* cwp = THLongTensor_data(conWei);
  const Real* gop = Tensor_(data)(gradOutput);
  Real* gwp = Tensor_(data)(gradWeight);
  const long* orp  = THLongTensor_data(order);
  const long* orcp = THLongTensor_data(orderChk);

  /*
  int i, j, iChk;
  long sidx, didx;
#pragma omp parallel for private(iChk, i, j, sidx, didx)
  for (iChk = 0; iChk < nChunks; ++iChk) {
    const int idst = orcp[iChk+1];
    for (i = orcp[iChk]; i < idst; ++i) {
      const int k = orp[i];
      sidx = didx = 0;
      for (j = 0; j < nDims; ++j) {
	sidx += is [j] * csp[k*css0 + j];
	didx += gos[j] * cdp[k*cds0 + j];
      }
      gwp[cwp[k]*gws] += scale * ip[sidx] * gop[didx];
    }
  }
  */

  int i, j, iChk;
#pragma omp parallel for private(iChk, i, j)
  for (iChk = 0; iChk < nChunks; ++iChk) {
    const int idst = orcp[iChk+1];
    for (i = orcp[iChk]; i < idst; ++i) {
      const int k = orp[i];
      gwp[cwp[k]*gws] += scale * ip[is[0]*csp[k*css0+j]] * gop[gos[0]*cdp[k*cds0+j]];
    }
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

static int Sort(lua_State* L) {
  const char* idlong = "torch.LongTensor";
  THLongTensor* input  = (THLongTensor*)luaT_checkudata(L, 1, idlong);
  THLongTensor* output = (THLongTensor*)luaT_checkudata(L, 2, idlong);
  THLongTensor* idx    = (THLongTensor*)luaT_checkudata(L, 3, idlong);

  long n = input->size[0];
  long is = input->stride[0];
  long* ip = THLongTensor_data(input);
  long os = output->stride[0];
  long* op = THLongTensor_data(output);
  long xs = idx->stride[0];
  long* xp = THLongTensor_data(idx);
  vector<pair<long, long> > v(n);
  for (long i = 0; i < n; ++i)
    v[i] = pair<long, long>(ip[is*i], i);
  sort(v.begin(), v.end());
  for (long i = 0; i < n; ++i) {
    op[os*i] = v[i].first;
    xp[xs*i] = v[i].second;
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
  {"sort", Sort},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libhessian(lua_State *L) {
  luaL_openlib(L, "libhessian", libhessian, 0);
  return 1;
}
