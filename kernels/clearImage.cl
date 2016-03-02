#include "common.cl"

__kernel void clearImage(
    __write_only  image2d_t img)
{
	int2 gid = {get_global_id(0), get_global_id(1)};

	if(isWithin(gid, img)){
		write_imagef(img, gid, 0);
    }
}
