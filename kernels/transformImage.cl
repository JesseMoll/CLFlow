#include "common.cl"

__kernel void transformImage(
			__read_only image2d_t src,
			__read_only image2d_t uv_duv,
			float mult,
			__write_only image2d_t dst)
{
	float sigma = .5f;
	float two_sigma_squared = 2 * sigma * sigma;

    int2 posOut = {get_global_id(0), get_global_id(1)};
    float2 uv = 0;

    float2 posIn = convert_float2(posOut) - uv * mult;


    float4 pixel = 0;
    float totalWeight = 0;
    int2 offset;

    int size = two_sigma_squared * 2;
	for(offset.x = -size; offset.x <= size; offset.x++)
		for(offset.y = -size; offset.y <= size; offset.y++){
			float weight = 0;
			int2 pos = posOut + offset;

			//float2 dist =  posIn - pos;
			weight = native_exp(-(float)dot2i(offset, offset) / two_sigma_squared);


			if(isWithin(pos, src)){
				totalWeight += weight;
				uv += (read_imagef(uv_duv, samp, pos).xy + read_imagef(uv_duv, samp, pos).zw) * weight;//read_imagef(src, sampInterp, pos) * weight;
			}


		}
	uv /= totalWeight;
	pixel = read_imagef(src, sampInterp, convert_float2(posOut) - uv * mult + .5f);

	if(isWithin(posOut, dst)){
		write_imagef(dst, posOut, pixel);
    }
}
