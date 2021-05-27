all:
	nvcc -I. -arch=sm_35 -c src/splitstep.cu -o build/splitstep.o
	g++ -o NLSE.exe src/main.cpp build/splitstep.o -L/usr/local/cuda/lib64 -lcudart -L/usr/local/cuda/lib -lcufft

clean:
	@rm -rf *.o NLSE.exe build/splitstep
	@rm -rf *~
