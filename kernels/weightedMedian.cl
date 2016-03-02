#include "common.cl"

#define MEDIAN_WIDTH 6
#define LOCAL_SIZE 16
#define ARRAY_SIZE (MEDIAN_WIDTH * 2 + LOCAL_SIZE)

__kernel void weightedMedian(
		__read_only image2d_t uvIn,
		__read_only image2d_t occData,
		__write_only image2d_t uvOut)
{

	int2 gid = {get_global_id(0), get_global_id(1)};
	int2 lid = {get_local_id(0), get_local_id(1)};

	const float sigmaX = 7;
	const float sigmaX_2 = 2 * sigmaX * sigmaX;

	const float sigmaI = 21;
	const float sigmaI_2 = 2 * sigmaI * sigmaI;

	__local float2 lInput [ARRAY_SIZE * ARRAY_SIZE]; //avoid memory bank collisions by storing float2 instead of float4
	__local float2 lUV [ARRAY_SIZE * ARRAY_SIZE];


	int2 startPos = gid - lid - MEDIAN_WIDTH;
	for(int x = lid.x; x < ARRAY_SIZE; x += LOCAL_SIZE){
		for(int y = lid.y; y < ARRAY_SIZE; y += LOCAL_SIZE) {
			float4 readData = read_imagef(occData, sampBorder, startPos + (int2)(x,y));
			int index = x + y * ARRAY_SIZE;
			lInput[index] = readData.xy;
			lUV[index] = readData.zw;

		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	float midVal = lInput[ (lid.x + MEDIAN_WIDTH) + (lid.y + MEDIAN_WIDTH) * ARRAY_SIZE].y;

	float2 maxUV = -HUGE_VALF;
	float2 minUV =  HUGE_VALF;

	int2 lPos;
	for(lPos.x = 0; lPos.x <= MEDIAN_WIDTH * 2; lPos.x++){
		for(lPos.y = 0; lPos.y <= MEDIAN_WIDTH * 2; lPos.y++) {


			int index = lid.x + lPos.x + (lid.y + lPos.y) * ARRAY_SIZE;
			float2 uv = lUV[index];
			maxUV = max(maxUV, uv);
			minUV = min(minUV, uv);
		}
	}


	float maxRange = max(maxUV.x - minUV.x, maxUV.y - minUV.y);


	const int iters = min((int)(native_log2(maxRange * 128)), 8);
	for (int itr = 0; itr < iters; itr++){

		float2 balance = 0; //higher or lower than mid?
		for(lPos.x = 0; lPos.x <= MEDIAN_WIDTH * 2; lPos.x++){
#pragma unroll
			for(lPos.y = 0; lPos.y <= MEDIAN_WIDTH * 2; lPos.y++) {

				int index = lid.x + lPos.x + (lid.y + lPos.y) * ARRAY_SIZE;

				float2 uv = lUV[index];
				float2 occ_val = lInput[index];

				float dif = occ_val.y - midVal;

				int distSquared = dot2i(lPos - MEDIAN_WIDTH, lPos - MEDIAN_WIDTH);
				float factor = 		-dif*dif 		/ (sigmaI_2) 	//intensity
											-distSquared 	/ (sigmaX_2);	//spatial

				float weight = native_exp(factor) * occ_val.x;

				//if(lPos.x == MEDIAN_WIDTH && lPos.y == MEDIAN_WIDTH) continue;

				balance += select((float2)weight, -weight, isless(uv, (minUV + maxUV) / 2));
			}
		}

		//		midAdder /=2;
		//		mid = select(mid + midAdder, mid - midAdder, isgreater(balance, 0));

		int2 comp = isgreater(balance, 0);

		float2 tmp = (minUV + maxUV)/2;
		minUV = select(minUV, tmp, comp);
		maxUV = select(tmp, maxUV, comp);


	}



	//Instead of using the value, we can interpolate between 2 closest to the mid
	float2 mid = (minUV + maxUV) / 2;
//	maxUV = -HUGE_VALF;
//	minUV = HUGE_VALF;
//
//	for(lPos.x = 0; lPos.x <= MEDIAN_WIDTH * 2; lPos.x++){
//		for(lPos.y = 0; lPos.y <= MEDIAN_WIDTH * 2; lPos.y++) {
//
//			int index = lid.x + lPos.x + (lid.y + lPos.y) * ARRAY_SIZE;
//			float2 uv = lUV[index];
//			maxUV = select(maxUV, max(maxUV, uv), isless(uv, mid));
//			minUV = select(minUV, min(minUV, uv), isgreater(uv, mid));
//		}
//	}
//	mid = (minUV + maxUV) / 2;

	float4 oldUV = read_imagef(uvIn, sampBorder, gid);



	float4 outputVal;
	outputVal.xy = oldUV.xy;
	outputVal.zw = mid - oldUV.xy;



	if(isWithin(gid, uvOut))
		write_imagef(uvOut, gid, outputVal);
}
