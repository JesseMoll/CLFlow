#include "common.cl"

__kernel void copyImage(
    __read_only  image2d_t src,
    __write_only image2d_t dst)
{
	int2 gid = {get_global_id(0), get_global_id(1)};


	float2 dstSize = {get_image_width(dst), get_image_height(dst)};

	if(gid.x < dstSize.x && gid.y < dstSize.y){
		float4 pixel = read_imagef(src, sampBorder, gid);
		write_imagef(dst, gid, pixel);
    }
}
