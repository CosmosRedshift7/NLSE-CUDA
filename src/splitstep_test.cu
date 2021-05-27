%%writefile src/splitstep.cu
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include "helper_cuda.h"

const int BLOCKDIM = 32;
//---------------------------------------------------------------------------
template<typename T>
std::vector<T> linspace(T start_in, T end_in, int num_in)
{
  std::vector<T> linspaced;
  T start = static_cast<T>(start_in);
  T end = static_cast<T>(end_in);
  T num = static_cast<T>(num_in);
  if (num == 0) { return linspaced; }
  if (num == 1)
  {
    linspaced.push_back(start);
    return linspaced;
  }
  T delta = (end - start) / (num - 1);
  for(int i=0; i < num-1; ++i) linspaced.push_back(start + delta * i);
  linspaced.push_back(end);
  return linspaced;
}
//---------------------------------------------------------------------------
template<typename T>
std::vector<T> arange(T start, T stop, T step = 1)
{
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}
//---------------------------------------------------------------------------
template<typename T>
__global__ void create_linear_propagator(cufftDoubleComplex* LP, T* w, T d, T z_step, int dim_t)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	LP[i] = make_cuDoubleComplex(cos(d*z_step/2*w[i]*w[i]), -sin(d*z_step/2*w[i]*w[i]));
  LP[i].x = LP[i].x/dim_t;
  LP[i].y = LP[i].y/dim_t;
}
//---------------------------------------------------------------------------
template<typename T>
__global__ void E_input(cufftDoubleComplex* E0, T* t, int* a, int seq_len, T pulse_width)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

  E0[i] = make_cuDoubleComplex(0, 0);
  cufftDoubleComplex E0_k;
  int half_seq_len = (seq_len - 1) / 2;
  for (int k = 0; k < seq_len; k++)
  {
    E0_k = make_cuDoubleComplex(a[k]/(pow(M_PI, 0.25))*exp(-0.5*pow((t[i] - (k - half_seq_len)*pulse_width), 2)), 0);
    E0[i] = cuCadd(E0[i], E0_k);
  }
}
//---------------------------------------------------------------------------
__global__ void half_lin_mul(cufftDoubleComplex *u, cufftDoubleComplex *LP)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	u[i] = cuCmul(u[i], LP[i]);
}

void half_lin(cufftDoubleComplex *u, cufftDoubleComplex *LP, cufftHandle plan, dim3 grid, dim3 block)
{
  cufftExecZ2Z(plan, u, u, CUFFT_FORWARD);
	half_lin_mul<<<grid, block>>>(u, LP);
  cufftExecZ2Z(plan, u, u, CUFFT_INVERSE);
}
//---------------------------------------------------------------------------
template<typename T>
__global__ void nonlin(cufftDoubleComplex *u, T nonlinearity, T z_step)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  T phi = nonlinearity*z_step*cuCabs(u[i])*cuCabs(u[i]);
  u[i] = cuCmul(u[i], make_cuDoubleComplex(cos(phi), sin(phi)));
}
//---------------------------------------------------------------------------

void split_step_gpu(int seq_len,
										int dim_t,
        						double dispersion,
        						double nonlinearity,
        						double pulse_width,
        						double z_end,
						        double z_step,
						        bool disp_compensate)
{
	double t_end, tMax, tMin, t_step, dw;

	std::vector<int> a(seq_len);
	std::vector<double> t;
	std::vector<double> w;
	std::vector<double> w2;
  std::vector<double> output(2*dim_t);

  std::ofstream output_file;

  // int dim_z;
	// dim_z = int(z_end / z_step) + 1;
	// std::vector<double> z;
	// z = linspace(double(0), z_end, dim_z);
	//-------------------print test-------------------
	// std::cout << "z = ";
	// for (auto n : z) std::cout << n << " ";
	// std::cout << "\n";
	//------------------------------------------------
	t_end = (int((seq_len - 1)/2) + 1) * pulse_width;
	tMax = t_end + 5*sqrt(2*(1 + z_end*z_end));
	tMin = -tMax;
	t_step = (tMax - tMin) / dim_t;
	t = linspace(tMin, tMax-t_step, dim_t);
	//-------------------print test-------------------
	// std::cout << "t = ";
	// for (auto n : t) std::cout << n << " ";
	// std::cout << "\n\n";
	//------------------------------------------------

	// prepare frequencies
	dw = 2 * M_PI / (tMax - tMin);
	w = arange(double(0), double(dim_t / 2 + 1));
	w2 = arange(double(-dim_t / 2 + 1), double(0));
	w.insert(w.end(), w2.begin(), w2.end());
	std::transform(w.begin(), w.end(), w.begin(), [&dw](auto& c){return c*dw;}); // w = dw * w
	//-------------------print test-------------------
	// std::cout << "w*dw = ";
	// for (auto n : w) std::cout << n << " ";
	// std::cout << "\n\n";
	//------------------------------------------------

	// Set initial condition
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_int_distribution<int> uni(1, 2);
	auto gen = [&uni, &rng](){return uni(rng);};
	std::generate(begin(a), end(a), gen);
	std::transform(a.begin(), a.end(), a.begin(), [&](auto& c){return 2*c - 3;}); // a = 2*a - 3
	//-------------------print test-------------------
	std::cout << "a = ";
	for (auto n : a) std::cout << n << " ";
	std::cout << "\n\n";
	//------------------------------------------------
	double *d_t, *d_w;
	int *d_a;
	cufftDoubleComplex *d_LP, *d_u;
  cufftHandle plan;

  checkCudaErrors(cudaMalloc(&d_a, sizeof(int)*seq_len));
  checkCudaErrors(cudaMalloc(&d_t, sizeof(double)*dim_t));
	checkCudaErrors(cudaMalloc(&d_w, sizeof(double)*dim_t));
  checkCudaErrors(cudaMalloc(&d_u, sizeof(cufftDoubleComplex)*dim_t));
	checkCudaErrors(cudaMalloc(&d_LP, sizeof(cufftDoubleComplex)*dim_t));

  checkCudaErrors(cudaMemcpy(d_a, a.data(), sizeof(int)*seq_len, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_t, t.data(), sizeof(double)*dim_t, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_w, w.data(), sizeof(double)*dim_t, cudaMemcpyHostToDevice));

  checkCudaErrors(cufftPlan1d(&plan, dim_t, CUFFT_Z2Z, 1));

	//---------------------------------------------------------------------------
	const dim3 block(BLOCKDIM);
	const dim3 grid((dim_t - 1) / BLOCKDIM + 1);

	// prepare linear propagator
	create_linear_propagator<<<grid, block>>>(d_LP, d_w, dispersion, z_step, dim_t);
	//-------------------print test-------------------
	// std::vector<double> LP(2*dim_t);
	// checkCudaErrors(cudaDeviceSynchronize());
	// checkCudaErrors(cudaMemcpy(LP.data(), d_LP, sizeof(double)*2*dim_t, cudaMemcpyDeviceToHost));
	// std::cout << "LP = ";
	// for (auto n : LP) std::cout << n << " ";
	// std::cout << "\n\n";
	//------------------------------------------------

	// Set initial condition
  E_input<<<grid, block>>>(d_u, d_t, d_a, seq_len, pulse_width);
  //-------------------print test-------------------
  // std::vector<double> E_z0(2*dim_t);
  // checkCudaErrors(cudaDeviceSynchronize());
  // checkCudaErrors(cudaMemcpy(E_z0.data(), d_u, sizeof(double)*2*dim_t, cudaMemcpyDeviceToHost));
  // std::cout << "E0 = ";
  // for (auto n : E_z0) std::cout << n << " ";
  // std::cout << "\n\n";
  //------------------------------------------------

  for (int i = 0; i < int(z_end/z_step); i++)
  {
    half_lin(d_u, d_LP, plan, grid, block);
    nonlin<<<grid, block>>>(d_u, nonlinearity, z_step);
    half_lin(d_u, d_LP, plan, grid, block);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(output.data(), d_u, sizeof(double)*2*dim_t, cudaMemcpyDeviceToHost));
  //-------------------print test-------------------
  // std::cout << "output = ";
  // for (auto n : output) std::cout << n << " ";
  // std::cout << "\n\n";
  //------------------------------------------------
  output_file.open("output.txt");
  for (auto n : output) output_file << n << " ";
  output_file.close();

  output_file.open("time.txt");
  for (auto n : t) output_file << n << " ";
  output_file.close();
	//---------------------------------------------------------------------------
  checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_t));
	checkCudaErrors(cudaFree(d_w));
	checkCudaErrors(cudaFree(d_u));
  checkCudaErrors(cudaFree(d_LP));
}
