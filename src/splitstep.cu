#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>
#include "helper_cuda.h"

const int BLOCKDIM = 1024;
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
  if (i >= dim_t) return;

	LP[i] = make_cuDoubleComplex(cos(d*z_step/2*w[i]*w[i]), -sin(d*z_step/2*w[i]*w[i]));
  // scale
  LP[i].x = LP[i].x/dim_t;
  LP[i].y = LP[i].y/dim_t;
}
//---------------------------------------------------------------------------
template<typename T>
__global__ void E_input(cufftDoubleComplex* E0, T* t, int* a, int seq_len, T pulse_width, int dim_t)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= dim_t) return;

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
__global__ void half_lin_mul(cufftDoubleComplex *u, cufftDoubleComplex *LP, int dim_t)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= dim_t) return;

	u[i] = cuCmul(u[i], LP[i]);
}

void half_lin(cufftDoubleComplex *u, cufftDoubleComplex *LP, cufftHandle plan, dim3 grid, dim3 block, int dim_t)
{
  cufftExecZ2Z(plan, u, u, CUFFT_FORWARD);
	half_lin_mul<<<grid, block>>>(u, LP, dim_t);
  cufftExecZ2Z(plan, u, u, CUFFT_INVERSE);
}
//---------------------------------------------------------------------------
template<typename T>
__global__ void nonlin(cufftDoubleComplex *u, T nonlinearity, T z_step, int dim_t)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= dim_t) return;

  T phi = nonlinearity*z_step*cuCabs(u[i])*cuCabs(u[i]);
  u[i] = cuCmul(u[i], make_cuDoubleComplex(cos(phi), sin(phi)));
}
//---------------------------------------------------------------------------
template<typename T>
__global__ void back_prop_mul(cufftDoubleComplex *u, T *w, T d, T z_end, int dim_t)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= dim_t) return;

  T phi = d*z_end*w[i]*w[i];
  u[i] = cuCmul(u[i], make_cuDoubleComplex(cos(phi), sin(phi)));
  // scale
  u[i].x = u[i].x/dim_t;
  u[i].y = u[i].y/dim_t;
}

template<typename T>
void back_propagation(cufftDoubleComplex *u, T *w, T d, T z_end, cufftHandle plan, dim3 grid, dim3 block, int dim_t)
{
  cufftExecZ2Z(plan, u, u, CUFFT_FORWARD);
	back_prop_mul<<<grid, block>>>(u, w, d, z_end, dim_t);
  cufftExecZ2Z(plan, u, u, CUFFT_INVERSE);
}
//---------------------------------------------------------------------------
__global__ void scale(cufftDoubleComplex *u, int dim_t)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= dim_t) return;

  u[i].x = u[i].x/dim_t;
  u[i].y = u[i].y/dim_t;
}
//---------------------------------------------------------------------------
void save_signal(cufftDoubleComplex *d_u, std::string fileName, int dim_t)
{
  std::vector<double> output(2*dim_t);
  std::ofstream output_file;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(output.data(), d_u, sizeof(double)*2*dim_t, cudaMemcpyDeviceToHost));
  output_file.open(fileName);
  for (auto n : output) output_file << n << " ";
  output_file.close();
}

void save_spectr(cufftDoubleComplex *d_u, std::string fileName, cufftHandle plan, dim3 grid, dim3 block, int dim_t)
{
  cufftExecZ2Z(plan, d_u, d_u, CUFFT_FORWARD);
  save_signal(d_u, fileName, dim_t);
  cufftExecZ2Z(plan, d_u, d_u, CUFFT_INVERSE);
  scale<<<grid, block>>>(d_u, dim_t);
}
//---------------------------------------------------------------------------
void split_step_gpu(int    seq_len,
										int    dim_t,
        						double dispersion,
        						double nonlinearity,
        						double pulse_width,
        						double z_end,
						        double z_step)
{
	double t_end, tMax, tMin, t_step, dw;

	std::vector<int> a(seq_len);
	std::vector<double> t;
	std::vector<double> w;
	std::vector<double> w2;
  std::ofstream output_file;
  std::string saveDir = "results/";
  // prepare time points
	t_end = (int((seq_len - 1)/2) + 1) * pulse_width;
	tMax = t_end + 2*sqrt(2*(1 + z_end*z_end));
	tMin = -tMax;
	t_step = (tMax - tMin) / dim_t;
	t = linspace(tMin, tMax-t_step, dim_t);
	// prepare frequencies
	dw = 2 * M_PI / (tMax - tMin);
	w = arange(double(0), double(dim_t / 2 + 1));
	w2 = arange(double(-dim_t / 2 + 1), double(0));
	w.insert(w.end(), w2.begin(), w2.end());
	std::transform(w.begin(), w.end(), w.begin(), [&dw](auto& c){return c*dw;}); // w = dw * w
	// prepare amplitudes
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_int_distribution<int> uni(1, 2);
	auto gen = [&uni, &rng](){return uni(rng);};
	std::generate(begin(a), end(a), gen);
	std::transform(a.begin(), a.end(), a.begin(), [&](auto& c){return 2*c - 3;}); // a = 2*a - 3
  //---------------------------------------------------------------------------
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
	// set initial condition
  E_input<<<grid, block>>>(d_u, d_t, d_a, seq_len, pulse_width, dim_t);
  save_signal(d_u, saveDir + "input.txt", dim_t);
  save_spectr(d_u, saveDir + "input_spectr.txt", plan, grid, block, dim_t);

  // numerical integration (split-step)
  for (int i = 0; i < int(z_end/z_step); i++)
  {
    half_lin(d_u, d_LP, plan, grid, block, dim_t);
    nonlin<<<grid, block>>>(d_u, nonlinearity, z_step, dim_t);
    half_lin(d_u, d_LP, plan, grid, block, dim_t);
  }

  save_signal(d_u, saveDir + "output.txt", dim_t);
  save_spectr(d_u, saveDir + "output_spectr.txt", plan, grid, block, dim_t);

  // back propagation
  back_propagation(d_u, d_w, dispersion, z_end, plan, grid, block, dim_t);
  save_signal(d_u, saveDir + "output_back.txt", dim_t);
  save_spectr(d_u, saveDir + "output_back_spectr.txt", plan, grid, block, dim_t);

  // time and frequency save
  output_file.open(saveDir + "time.txt");
  for (auto n : t) output_file << n << " ";
  output_file.close();
  output_file.open(saveDir + "freq.txt");
  for (auto n : w) output_file << n << " ";
  output_file.close();
	//---------------------------------------------------------------------------
  checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_t));
	checkCudaErrors(cudaFree(d_w));
	checkCudaErrors(cudaFree(d_u));
  checkCudaErrors(cudaFree(d_LP));
}
