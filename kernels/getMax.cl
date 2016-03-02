#include "common.cl"

__kernel void getMax(
		__read_only image2d_t img,
		__global int* output)
{
	int2 gid = {get_global_id(0), get_global_id(1)};

	float4 val = read_imagef(img, sampBorder, gid);
	int4 iVal = convert_int4(fabs(val) * 1000);


	if(isWithin(gid, img)){
		atomic_max(&output[0], iVal.x);
		atomic_max(&output[1], iVal.y);
		atomic_max(&output[2], iVal.z);
		atomic_max(&output[3], iVal.w);
	}
}
