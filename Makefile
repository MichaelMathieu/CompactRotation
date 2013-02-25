all:
	g++ -fPIC -shared -O3 -fopenmp hessian.cpp -o libhessian.so -lgsl