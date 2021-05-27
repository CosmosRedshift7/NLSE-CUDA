#include <stdio.h>
#include <iostream>
#include <ctime>
#include <string>

extern void split_step_gpu(int 	  seq_len,
													 int 	  dim_t,
				        					 double dispersion,
				        					 double nonlinearity,
				        					 double pulse_width,
				        					 double z_end,
										       double z_step);

int main(int argc, char ** argv)
{
	int 	 seq_len 			= atoi(argv[1]);
	int 	 dim_t 				= atoi(argv[2]);
	double dispersion 	= atof(argv[3]);
	double nonlinearity = atof(argv[4]);
	double pulse_width 	= atof(argv[5]);
	double z_end 				= atof(argv[6]);
	double z_step 			= atof(argv[7]);

	clock_t t_0 = clock();
	split_step_gpu(seq_len,
								 dim_t,
								 dispersion,
								 nonlinearity,
								 pulse_width,
								 z_end,
								 z_step);
	std::cout << "Execution time: " << (double(clock()-t_0))/CLOCKS_PER_SEC  << " s.\n";

	return 0;
}
