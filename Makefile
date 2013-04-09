MKL_ROOT=/home/myrhev/local/intel/mkl
INCLUDE=-I${MKL_ROOT}/include
MKL_LINK_DIR=${MKL_ROOT}/lib/intel64
FLAGS=-fPIC -shared -O3 -fopenmp
LINK=-L${MKL_LINK_DIR} -lmkl_gnu_thread -lmkl_intel_ilp64 -lmkl_core
LINK = -Xlinker -start-group -Xlinker ${MKL_LINK_DIR}/libmkl_sequential.a -Xlinker ${MKL_LINK_DIR}/libmkl_gf_ilp64.a -Xlinker ${MKL_LINK_DIR}/libmkl_intel_ilp64.a -Xlinker ${MKL_LINK_DIR}/libmkl_core.a -Xlinker ${MKL_LINK_DIR}/libmkl_gnu_thread.a -Xlinker -end-group

all:
	g++ ${FLAGS} ${INCLUDE} hessian.cpp ${LINK} -o libhessian.so -lgsl