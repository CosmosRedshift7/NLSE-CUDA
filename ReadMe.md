# Solution of the nonlinear Schrödinger equation using cuFFT
High Performance Computing and Modern Architectures course
Final project
Ilya Kuk

## Prerequisites:
Make 'bulid' and 'results' folders in working dirrectory:
```shell
mkdir build results
```

## Running template:
```shell
./NLSE.exe <seq_len> <dim_t> <dispersion> <nonlinearity> <pulse_width> <z_end> <z_step>
```

## Example of running application:
```shell
./NLSE.exe 8 8192 0.5 0.05 10 100 0.1
```
