# Solution of the nonlinear Schrödinger equation using cuFFT
High Performance Computing and Modern Architectures course
Final
Ilya Kuk

## Running template:

`./NLSE.exe <seq_len> <dim_t> <dispersion> <nonlinearity> <pulse_width> <z_end> <z_step>`

## Example of running application:

`./NLSE.exe 8 8192 0.5 0.05 10 100 0.1`

## NLSE

Pulse propagation  in optical fiber links for moderate values of  power and spectral width of the pulse with high accuracy
governed by nonlinear Schrödinger
$$
iE_z + \frac{1}{2} E_{tt}+ \varepsilon |E|^2 E=0
$$

where $E=E(t,z)$ is dimensionless amplitude of the electric field, $t$ and $z$ are dimensionless retarded  time and distance, $\varepsilon$ is dimensionless coefficient of nonlinearity.  This equation is the result of spatial averaging of the original model with in-line  optical amplifiers periodically spaced for compensation of natural fiber losses

## Initial condition

Bit-sequence launched at  the front end of the system is represented by periodic train of gaussian pulses with Differential Phase Shift Keying (DPSK)
$$
E(t,0) = \sum\limits_{k=1}^{N} a_{k}\pi^{-1/4} \exp\left[-(t-kT)^2/2\right]
$$


## Split-step method


$$
E_z = i\frac{1}{2} E_{tt} + i \varepsilon |E|^2 E = [\hat{D} + \hat{N}]E
$$
The equation can be split into a linear part,
$$
E_z = i\frac{1}{2} E_{tt} = \hat{D} E
$$
and a nonlinear part,
$$
E_z = i \varepsilon |E|^2 E = \hat{N} E
$$

![](./images/spst.png)

* Half dispersion step

$$
E(t,z+\frac{dz}{2}) = F^{-1}(F[E(t,z)] \cdot \exp[-i \frac{1}{2}\frac{dz}{2}w^2]),
$$

where $F$ and $F^{-1}$ denotes forward and the reverse Fourier transform respectively.

* Nonlinear step

$$
E(t,z+dz) = E(t,z) \cdot \exp[i \varepsilon dz |E|^2]
$$



## Plots

<div align = "center"> <b>Input and back propagated signal intensity</b> </div>

![input](./images/input.svg)

<div align = "center"> <b>Output signal intensity</b> </div>

![input](./images/output.svg)

<div align = "center"> <b>Spectrum</b> </div>

![input](./images/spectrum.svg)

