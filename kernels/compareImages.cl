#include "common.cl"

__kernel void compareImages(
		__read_only image2d_t img1,
		__read_only image2d_t img2,

		__global int* output)
{
	barrier(CLK_LOCAL_MEM_FENCE);

	int2 gid = {get_global_id(0), get_global_id(1)};
	int2 lid = {get_local_id(0), get_local_id(1)};

	float3 i1 = read_imagef(img1, sampBorder, gid).xyz;
	float3 i2 = read_imagef(img2, sampBorder, gid).xyz;


	float d = distance(i1, i2) * 100.0f;
	int dI = (int) d;

	if(isWithin(gid, img1)){
		atomic_max(&output[0], dI);
		atomic_add(&output[1], dI);
	}

}
