#include "common.cl"

__kernel void resize(
    __read_only  image2d_t src,
    __write_only image2d_t dst)
{

	float2 srcSize = {get_image_width(src), get_image_height(src)};
	float2 dstSize = {get_image_width(dst), get_image_height(dst)};

	float2 scale = srcSize / dstSize;
	float sigma = native_sqrt(scale.x) / sqrt(3.0f);

	int2 posOut = {get_global_id(0), get_global_id(1)};
	float2 posIn = convert_float2(posOut) * scale;

	float4 pixel;
	if(srcSize.x == dstSize.x){
		//If image size is constant, just copy the values
		pixel = read_imagef(src, samp, posOut);
	}
	else if(srcSize.x > dstSize.x){
		//pixel = read_imagef(src, sampInterp, posIn + .5f);
		//When downscaling, perform gaussian interpolation
		pixel = readImage(src, posIn, sigma);
	}
	else{
		//When upscaling, just perform linear interpolation
		//pixel = read_imagef(src, sampInterp, posIn + .5f / scale);
		pixel = readImage(src, posIn, sigma);
	}

	//float4 pixel = readImage(src, posIn, sigma);

	if(isWithin(posOut, dst)){
		write_imagef(dst, posOut, pixel);
    }
}
