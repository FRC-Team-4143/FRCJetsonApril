
#include <cuda_runtime.h>
#include "raw2rgb.cuh"

__global__ void gpuConvertrawtoRGBA_kernel(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width) {
		return;
	}
	if((idx & 1) != 0) { // odd
		return;
	}

	for (int i = 0; i < height; i+=2) {
		unsigned char r  = (src[i*width+idx+0] );
		float g0 = (src[i*width+idx+1] );
		float g1 = (src[(i+1)*width+idx+0] ) ;
		unsigned char b  = (src[(i+1)*width+idx+1] );

		// green detection
		//if(g0 < 0x30 || r > 0x30 || b > 0x30) g0 = r = b = 0;
		//else { g0 = 0xff; r = b = 0;}
		
		dst[(i/2)*width/2*4+(idx/2)*4+0] = b;
		dst[(i/2)*width/2*4+(idx/2)*4+1] = ((g0 + g1)/2.5);
		dst[(i/2)*width/2*4+(idx/2)*4+2] = r;
		dst[(i/2)*width/2*4+(idx/2)*4+3] = 0;
	}
}
__global__ void gpuConvertgraytoRGBA_kernel(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width) {
		return;
	}

	for (int i = 0; i < height; i+=1) {
		unsigned char gray = (src[i*width+idx+0] );

		dst[i*width*4+idx*4+0] = gray;
		dst[i*width*4+idx*4+1] = gray;
		dst[i*width*4+idx*4+2] = gray;
		dst[i*width*4+idx*4+3] = 0;
	}
}
__global__ void gpuConvertgraytoRGB_kernel(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width) {
		return;
	}

	for (int i = 0; i < height; i+=1) {
		unsigned char gray = (src[i*width+idx+0] );

		dst[i*width*3+idx*3+0] = gray;
		dst[i*width*3+idx*3+1] = gray;
		dst[i*width*3+idx*3+2] = gray;
	}
}
__global__ void gpuConvertgraytoRGB_kernel(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	const int column = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	const int isrc = row*width + column;
	const int idst = row*width*3 + column*3;
	if ((column >= width) || (row >= height)) return;

	unsigned char gray = (src[isrc] ) >> 8;;

	dst[idst+0] = gray;
	dst[idst+1] = gray;
	dst[idst+2] = gray;
}
__global__ void gpuConvertgraytoRGBA_kernel(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width) {
		return;
	}
	if((idx & 1) != 0) { // odd
		return;
	}

	for (int i = 0; i < height; i+=2) {
		unsigned char r  = (src[i*width+idx+0] ) >> 2;
		unsigned char g0 = (src[i*width+idx+1] ) >> 2;
		unsigned char g1 = (src[(i+1)*width+idx+0] ) >> 2 ;
		unsigned char b  = (src[(i+1)*width+idx+1] ) >> 2;

		dst[i*width*4+idx*4+0] = b;
		dst[i*width*4+idx*4+1] = g0;
		dst[i*width*4+idx*4+2] = r;
		dst[i*width*4+idx*4+3] = 0;
		dst[i*width*4+idx*4+0+4] = b;
		dst[i*width*4+idx*4+1+4] = g0;
		dst[i*width*4+idx*4+2+4] = r;
		dst[i*width*4+idx*4+3+4] = 0;
		dst[(i+1)*width*4+idx*4+0] = b;
		dst[(i+1)*width*4+idx*4+1] = g0;
		dst[(i+1)*width*4+idx*4+2] = r;
		dst[(i+1)*width*4+idx*4+3] = 0;
		dst[(i+1)*width*4+idx*4+0+4] = b;
		dst[(i+1)*width*4+idx*4+1+4] = g0;
		dst[(i+1)*width*4+idx*4+2+4] = r;
		dst[(i+1)*width*4+idx*4+3+4] = 0;
	}
}
__global__ void gpuConvertrawtoRGB_kernel(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width) {
		return;
	}
	if((idx & 1) != 0) { // odd
		return;
	}

	for (int i = 0; i < height; i+=2) {
		unsigned char r  = (src[i*width+idx+0] );
		unsigned char g0 = (src[i*width+idx+1] );
		unsigned char g1 = (src[(i+1)*width+idx+0] ) ;
		unsigned char b  = (src[(i+1)*width+idx+1] );

		// green detection
		//if(g0 < 0x30 || r > 0x30 || b > 0x30) g0 = r = b = 0;
		//else { g0 = 0xff; r = b = 0;}
		
		dst[(i/2)*width/2*3+(idx/2)*3+0] = b;
		dst[(i/2)*width/2*3+(idx/2)*3+1] = ((g0 + g1)/2);
		dst[(i/2)*width/2*3+(idx/2)*3+2] = r;
	}
}

__global__ void gpuConvertrawtoRGB_kernel(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width) {
		return;
	}
	if((idx & 1) != 0) { // odd
		return;
	}

	for (int i = 0; i < height; i+=2) {
		unsigned char r  = src[i*width+idx+0] >> 7;
		unsigned char g0 = src[i*width+idx+1] >> 8;
		unsigned char g1 = src[(i+1)*width+idx+0] >> 9;
		unsigned char b  = src[(i+1)*width+idx+1] >> 7;

		// green detection
		//if(g0 < 0x30 || r > 0x30 || b > 0x30) g0 = r = b = 0;
		//else { g0 = 0xff; r = b = 0;}
		
		dst[(i/2)*width/2*3+(idx/2)*3+0] = b;
		dst[(i/2)*width/2*3+(idx/2)*3+1] = g0 + g1;
		dst[(i/2)*width/2*3+(idx/2)*3+2] = r;
	}
}
void gpuConvertrawtoRGB(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	unsigned short *d_src = NULL;
	unsigned char *d_dst = NULL;

	d_src = src;
	cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);

	d_dst = dst;
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);

	unsigned int blockSize = 1024;
	//unsigned int numBlocks = (width / 2 + blockSize - 1) / blockSize;
	unsigned int numBlocks = (width + blockSize - 1) / blockSize;
	gpuConvertrawtoRGB_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, width, height);
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}
void gpuConvertrawtoRGB(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	unsigned char *d_src = NULL;
	unsigned char *d_dst = NULL;

	d_src = src;
	cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);

	d_dst = dst;
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);

	unsigned int blockSize = 1024;
	//unsigned int numBlocks = (width / 2 + blockSize - 1) / blockSize;
	unsigned int numBlocks = (width + blockSize - 1) / blockSize;
	gpuConvertrawtoRGB_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, width, height);
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}
void gpuConvertrawtoRGBA(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	unsigned char *d_src = NULL;
	unsigned char *d_dst = NULL;

	d_src = src;
	cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);

	d_dst = dst;
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);

	unsigned int blockSize = 1024;
	//unsigned int numBlocks = (width / 2 + blockSize - 1) / blockSize;
	unsigned int numBlocks = (width + blockSize - 1) / blockSize;
	gpuConvertrawtoRGBA_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, width, height);
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}

void gpuConvertgraytoRGBA(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	unsigned char *d_src = NULL;
	unsigned char *d_dst = NULL;

	d_src = src;
	cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);

	d_dst = dst;
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);

	unsigned int blockSize = 1024;
	unsigned int numBlocks = (width + blockSize - 1) / blockSize;
	gpuConvertgraytoRGBA_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, width, height);
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}

void gpuConvertgraytoRGB(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	unsigned char *d_src = NULL;
	unsigned char *d_dst = NULL;

	d_src = src;
	cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);

	d_dst = dst;
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);

	unsigned int blockSize = 1024;
	unsigned int numBlocks = (width + blockSize - 1) / blockSize;
	gpuConvertgraytoRGB_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, width, height);
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}

void gpuConvertgraytoRGB(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height, cudaStream_t stream)
{
	unsigned short *d_src = NULL;
	unsigned char *d_dst = NULL;

	d_src = src;
	cudaStreamAttachMemAsync(stream, src, 0, cudaMemAttachGlobal);

	d_dst = dst;
	cudaStreamAttachMemAsync(stream, dst, 0, cudaMemAttachGlobal);

	//unsigned int blockSize = 128;
	//unsigned int numBlocks = (width + blockSize - 1) / blockSize;
	//gpuConvertgraytoRGB_kernel<<<numBlocks, blockSize, 0, stream>>>(d_src, d_dst, width, height);

	const dim3 blockSize(32,32);
	const int bx = (width + 32 - 1)/32;
	const int by = (height + 32 - 1)/32;

	const dim3 gridSize = dim3(bx, by);
	gpuConvertgraytoRGB_kernel<<<gridSize, blockSize, 0, stream>>>(d_src, d_dst, width, height);



	//cudaStreamAttachMemAsync(stream, dst, 0, cudaMemAttachHost);
	//cudaStreamSynchronize(stream);
}

void gpuConvertgraytoRGBA(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	unsigned short *d_src = NULL;
	unsigned char *d_dst = NULL;

	d_src = src;
	cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);

	d_dst = dst;
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);

	unsigned int blockSize = 1024;
	unsigned int numBlocks = (width + blockSize - 1) / blockSize;
	gpuConvertgraytoRGBA_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, width, height);
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);
}
