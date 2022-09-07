//reference http://tecsingularity.com/cuda/cudaprogram1/

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <time.h>
#include <cuda.h>

bool to_bool(const char* str)
{
    std::istringstream is( str );
    bool result;
    is >> std::boolalpha >> result;
    return result;
}


// GPUで計算する際の関数
__global__ void gpu_function(float *d_x, float *d_y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_y[i] = sin(d_x[i]) * sin(d_x[i]) + cos(d_x[i]) * cos(d_x[i]);
}

// CPUで計算する際の関数
void cpu_function(int n, float *x, float *y)
{
	for (int i = 0; i < n; i++) {
		y[i] = sin(x[i]) * sin(x[i]) + cos(x[i]) * cos(x[i]);
	}
}

// main function
int main(int argc, char** argv)
{
	if(argc < 2){
		std::cout << "Usage: ./test1 [true/false]" << std::endl;
		return 0;
	}

	bool GPU = to_bool(argv[1]);
	if(GPU){
		printf("GPU mode\n");
	}else{
		printf("CPU mode\n");
	}

	int N = 100000000;
	float *host_x, *host_y, *dev_x, *dev_y;
	
	// CPU側の領域確保
	host_x = (float*)malloc(N * sizeof(float));
	host_y = (float*)malloc(N * sizeof(float));

	// 乱数値を入力する
	for (int i = 0; i < N; i++) {
		host_x[i] = rand();
	}

	int start = clock();

	if (GPU == true) {

		// デバイス(GPU)側の領域確保
		cudaMalloc(&dev_x, N * sizeof(float));
		cudaMalloc(&dev_y, N * sizeof(float));

		// CPU⇒GPUのデータコピー
		cudaMemcpy(dev_x, host_x, N * sizeof(float), cudaMemcpyHostToDevice);

		// GPUで計算
		gpu_function << <(N + 255) / 256, 256 >> >(dev_x, dev_y);

		// GPU⇒CPUのデータコピー
		cudaMemcpy(host_y, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	}
	else {
		// CPUで計算
		cpu_function(N, host_x, host_y);
	}

	int end = clock();

	// 計算が正しく行われているか確認
	float sum = 0.0f;
	for (int j = 0; j < N; j++) {
		sum += host_y[j];
	}
	std::cout << sum << std::endl;

	// 最後に計算時間を表示
	std::cout << "clock : "<< (end - start) << std::endl;

	return 0;
}