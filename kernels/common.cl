__constant int2 UP = {0, -1};
__constant int2 DOWN = {0, 1};
__constant int2 LEFT = {-1, 0};
__constant int2 RIGHT = {1, 0};

__constant float2 UPf = {0, -1};
__constant float2 DOWNf = {0, 1};
__constant float2 LEFTf = {-1, 0};
__constant float2 RIGHTf = {1, 0};

__constant sampler_t samp =
			CLK_NORMALIZED_COORDS_FALSE |
			CLK_ADDRESS_CLAMP_TO_EDGE |
			CLK_FILTER_NEAREST;

__constant sampler_t sampInterp =
			CLK_NORMALIZED_COORDS_FALSE |
			CLK_ADDRESS_CLAMP_TO_EDGE |
			CLK_FILTER_LINEAR;


__constant sampler_t sampInterpBorder =
			CLK_NORMALIZED_COORDS_FALSE |
			CLK_ADDRESS_CLAMP |
			CLK_FILTER_LINEAR;

__constant sampler_t sampBorder =
			CLK_NORMALIZED_COORDS_FALSE |
			CLK_ADDRESS_CLAMP |
			CLK_FILTER_NEAREST;

__constant sampler_t sampExact =
			CLK_NORMALIZED_COORDS_FALSE |
			CLK_ADDRESS_NONE |
			CLK_FILTER_NEAREST;



int dot2i(int2 a, int2 b){
	return a.x * b.x + a.y * b.y;
}

short dot2s(short2 a, short2 b){
	return a.x * b.x + a.y * b.y;
}


bool isWithin(int2 coord, image2d_t im){
	if (coord.x >= get_image_width(im))
		return false;
	if (coord.y >= get_image_height(im))
		return false;
	if (min(coord.x, coord.y) < 0)
		return false;
	return true;
}

bool isWithinF(float2 coord, image2d_t im){
	if (coord.x >= get_image_width(im))
		return false;
	if (coord.y >= get_image_height(im))
		return false;
	if (min(coord.x, coord.y) < 0)
		return false;
	return true;
}

float sqr(float x){
	return x*x;
}




float4 readImage(image2d_t img, float2 readPos, float sigma){

	//int2 readPosI = convert_int2(readPos);
	float two_sigma_squared = 2 * sigma * sigma;

	float4 pixel = 0;
	float totalWeight = .00001f;
	int2 offset;

	int size = convert_int_rte(2.5f*sigma);
	//int size = convert_int_rtp(two_sigma_squared * 8.0f);
	if(sigma == 0){
		return read_imagef(img, sampInterp, readPos + .5f);
	}

	for(offset.x = -size; offset.x <= size; offset.x++)
		for(offset.y = -size; offset.y <= size; offset.y++){
			float weight = 0;
			float2 pos = readPos + convert_float2(offset)/2;

			float2 dist =  readPos - convert_float2(pos);
			weight = native_exp(-dot(dist,dist) / two_sigma_squared);


			if(isWithinF(pos, img)){
				totalWeight += weight;
				pixel += read_imagef(img, samp, pos) * weight;
			}


		}
	pixel /= totalWeight;

	return pixel;
}
