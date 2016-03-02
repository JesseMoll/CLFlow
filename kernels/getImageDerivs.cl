#include "common.cl"

constant float filter[5] = {1.0f / 12.0f, -8.0f / 12.0f, 0, 8.0f / 12.0f, -1.0f / 12.0f};
//constant float filter[5] = {0, -0.5f, 0, 0.5f, 0};

float4 pdvRight(image2d_t input, int2 coord){
	float4 d = 0;

	#pragma unroll 5
	for(int x = 0; x < 5; x++){
		#pragma unroll 5
		for(int y = 0; y < 5; y++){
			int2 currCoord = coord - 2;
			currCoord.x += x;
			currCoord.y += y;
			float readVal = read_imagef(input, samp, currCoord).y;
			if(x == 2 && y == 2)
				d.s0 = readVal; //value
			if(y == 2)
				d.s1 += readVal * filter[x]; //dx
			if(x == 2)
				d.s2 += readVal * filter[y]; //dy
			d.s3 += readVal * filter[x] * filter[y]; //dxy
		}
	}
    return d;
}

float4 pdvLeft(image2d_t input, int2 coord){
	float4 d = 0;

	#pragma unroll 5
	for(int x = 0; x < 5; x++)
		#pragma unroll 5
		for(int y = 0; y < 5; y++){
			int2 currCoord = coord - 2;
			currCoord.x += x;
			currCoord.y += y;
			float readVal = read_imagef(input, samp, currCoord).x;
			if(x == 2 && y == 2)
				d.s0 = readVal; //value
			if(y == 2)
				d.s1 += readVal * filter[x]; //dx
			if(x == 2)
				d.s2 += readVal * filter[y]; //dy
			d.s3 += readVal * filter[x] * filter[y]; //dxy
	}
    return d;
}



__kernel void getImageDerivs(
			__read_only image2d_t LAB,
			__write_only image2d_t leftDerivs,
			__write_only image2d_t rightDerivs)
{
    int2 coord = {get_global_id(0), get_global_id(1)};

 	if(isWithin(coord, rightDerivs))
		write_imagef(rightDerivs, coord, pdvRight(LAB, coord));

 	if(isWithin(coord, leftDerivs))
		write_imagef(leftDerivs, coord, pdvLeft(LAB, coord));
}
