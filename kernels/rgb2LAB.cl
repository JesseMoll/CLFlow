#include "common.cl"

__kernel void rgb2LAB(
		__read_only image2d_t left,
		__read_only image2d_t right,
			//__global int* minMax,
			__write_only image2d_t output)
{
    const int2 coord = {get_global_id(0), get_global_id(1)};



    float sigma = sqrt(2.0f) / 2;

//    float3 rgbLeft = read_imagef(left, sampExact, coord).xyz;
//    float3 rgbRight = read_imagef(right, sampExact, coord).xyz;

	float3 rgbLeft = readImage(left, convert_float2(coord), sigma).xyz;
	float3 rgbRight = readImage(right, convert_float2(coord), sigma).xyz;


	float3 yVec = {0.333333333f, 0.333333333f, 0.333333333f};//{0.212671f, 0.715160f, 0.072169f};

	float2 y;

	y.s0 = dot(rgbLeft,yVec);
	y.s1 = dot(rgbRight,yVec);


	float2 y_1_3 = pow(y, 1.0f/3.0f);


	const float2 T = 0.008856f;


	float2 L = select(903.3f * y, 116 * y_1_3 - 16.0f, isgreater(y, T));


	L = clamp(L / 8, 0, 255);

	float4 outputVal = {L.s0, L.s1, 0, 0};

   	if(isWithin(coord, output))
		write_imagef(output, coord, outputVal);
}
