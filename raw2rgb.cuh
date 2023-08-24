/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __RAW2RGB_CUH__
#define __RAW2RGB_CUH__

void gpuConvertrawtoRGB(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height);
void gpuConvertrawtoRGBA(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height);
void gpuConvertrawtoRGB(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height);
void gpuConvertgraytoRGBA(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height);
void gpuConvertgraytoRGB(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height, cudaStream_t stream);
void gpuConvertgraytoRGB(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height, cudaStream_t stream);
void gpuConvertgraytoRGBA(unsigned short *src, unsigned char *dst,
		unsigned int width, unsigned int height);

#endif
