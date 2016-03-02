#include "common.cl"

__kernel void getEdges(
    __read_only  image2d_t src,
    __write_only image2d_t dst)
{
	int2 gid = {get_global_id(0), get_global_id(1)};

	float filter[3] = {1, 2, 1};
	
	float2 sobelX = 0;
	float2 sobelY = 0;
	
	
	for(int n = 0; n < 3; n++){
		float4 uv_duv = read_imagef(src, samp, gid + (int2)(n-1, -1));
		sobelX -= (uv_duv.xy + uv_duv.zw) * filter[n];
	}
	
	for(int n = 0; n < 3; n++){
		float4 uv_duv = read_imagef(src, samp, gid - (int2)(n-1, 1));
		sobelX += (uv_duv.xy + uv_duv.zw) * filter[n];
	}

	for(int n = 0; n < 3; n++){
		float4 uv_duv = read_imagef(src, samp, gid + (int2)(-1, n-1));
		sobelY -= (uv_duv.xy + uv_duv.zw) * filter[n];
	}
	
	for(int n = 0; n < 3; n++){
		float4 uv_duv = read_imagef(src, samp, gid + (int2)(1, n-1));
		sobelY += (uv_duv.xy + uv_duv.zw) * filter[n];
	}
	
	float2 edge = sobelX * sobelX + sobelY * sobelY;
	
	int output = max(edge.x, edge.y);

	if(isWithin(gid, dst)){
		write_imagei(dst, gid, output);
    }
}
