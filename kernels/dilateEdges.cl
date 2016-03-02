#include "common.cl"

__kernel void dilateEdges(
    __read_only  image2d_t src,
    __write_only image2d_t dst)
{
	int2 gid = {get_global_id(0), get_global_id(1)};

	int2 offset;


	int dsz = 2;
	int output = 0;
	for(offset.x = -dsz; offset.x <= dsz; offset.x++){
		for(offset.y = -dsz; offset.y <= dsz; offset.y++){
			output = max(output, read_imagei(src, samp, gid + offset).x);
		}
	}
	if(isWithin(gid, dst)){
		write_imagei(dst, gid, output);
    }
}
