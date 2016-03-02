#include "common.cl"


__kernel void detectOcclusion(
			__read_only image2d_t offset,
			__read_only image2d_t originalImage,
			__read_only image2d_t warpedImage, //{ZXI, ZYI, ZIt, ZI}
			__write_only image2d_t output)
{
	int2 coord = {get_global_id(0), get_global_id(1)};

	float right = read_imagef(offset, samp, coord + RIGHT).x + read_imagef(offset, samp, coord + RIGHT).z;
	float left = read_imagef(offset, samp, coord + LEFT).x + read_imagef(offset, samp, coord + LEFT).z;
	float up = read_imagef(offset, samp, coord + UP).y + read_imagef(offset, samp, coord + UP).w;
	float down = read_imagef(offset, samp, coord + DOWN).y + read_imagef(offset, samp, coord + DOWN).w;

	float2 uv;// = (float2)(right + left, up + down);
	uv = read_imagef(offset, samp, coord).xy + read_imagef(offset, samp, coord).zw;
	//uv /= 3;


	float org = read_imagef(originalImage, samp, coord).y;
	float4 I = read_imagef(warpedImage, samp, coord);


	float2 gradient = {right - left, down - up};


	if(coord.x == 0 || coord.x == get_image_width(offset) - 1)
		gradient.x /= 2;
	else
		gradient.x /= 3;
	if(coord.y == 0 || coord.y == get_image_height(offset) - 1)
		gradient.y /= 2;
	else
		gradient.y /= 3;

	//gradient /= 3;

	//gradient = I.xy;
	float div = min(min(gradient.x, gradient.y), 0.0f);//Divergence


//	float It =  I.w - org;
	float It = I.z;
//	It += dot(I.xy, uv);  //Linearize It

	const float sigmaD = .3f;
	const float sigmaD_2 = 2 * sigmaD * sigmaD;

	const float sigmaI = 20.0f;
	const float sigmaI_2 = 2 * sigmaI * sigmaI;

    float occ = native_exp(
    		-div*div / sigmaD_2		//Divergence
    		-It * It / sigmaI_2);	//Intensity

    if(isnan(I.w)){
    	occ = 0;
    }

    float4 valout = {occ, org, uv.x, uv.y};

    if(isWithin(coord, output))
    	write_imagef(output, coord, valout);
}
