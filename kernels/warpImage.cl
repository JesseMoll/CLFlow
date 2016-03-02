#include "common.cl"

__kernel void warpImage(
		__read_only image2d_t leftDerivs,
		__read_only image2d_t rightDerivs,
		__read_only image2d_t orig,
		__read_only image2d_t uv_duv,
		__write_only image2d_t Z)
{


	int2 writeCoord = {get_global_id(0), get_global_id(1)};

	float4 offset4 = read_imagef(uv_duv, sampBorder, writeCoord);
	float2 offset = offset4.xy + offset4.zw / 2;

	float2 readCoordF = convert_float2(writeCoord) + offset;

	int2 fXY = convert_int2_rtn(readCoordF);
	int2 cXY = fXY + 1;

	float2 alpha = readCoordF - convert_float2(fXY);

//	bool oob = ( cXY.x >= get_image_width(rightDerivs) ||
//			cXY.y >= get_image_height(rightDerivs) ||
//			fXY.x < 0 ||
//			fXY.y < 0);


	//{value, dx, dy, dxy} previously calculated

	float4 d00 = read_imagef(rightDerivs, sampBorder, fXY);
	float4 d01 = read_imagef(rightDerivs, sampBorder, (int2)(fXY.x, cXY.y));
	float4 d10 = read_imagef(rightDerivs, sampBorder, (int2)(cXY.x, fXY.y));
	float4 d11 = read_imagef(rightDerivs, sampBorder, cXY);

	float16 C;
	C.s0  =  1 * d00.s0;
	C.s1  =  1 * d00.s2;
	C.s2  = -3 * d00.s0 + 3 * d01.s0 - 2 * d00.s2 - 1 * d01.s2;
	C.s3  =  2 * d00.s0 - 2 * d01.s0 + 1 * d00.s2 + 1 * d01.s2;

	C.s4  =  1 * d00.s1;
	C.s5  =  1 * d00.s3;
	C.s6  = -3 * d00.s1 + 3 * d01.s1 - 2 * d00.s3 - 1 * d01.s3;
	C.s7  =  2 * d00.s1 - 2 * d01.s1 + 1 * d00.s3 + 1 * d01.s3;

	C.s8  = -3 * d00.s0 + 3 * d10.s0 - 2 * d00.s1 - 1 * d10.s1;
	C.s9  = -3 * d00.s2 + 3 * d10.s2 - 2 * d00.s3 - 1 * d10.s3;

	C.sA =   9 * d00.s0 - 9 * d10.s0 + 9 * d11.s0 - 9 * d01.s0 +
			6 * d00.s1 + 3 * d10.s1 - 3 * d11.s1 - 6 * d01.s1 +
			6 * d00.s2 - 6 * d10.s2 - 3 * d11.s2 + 3 * d01.s2 +
			4 * d00.s3 + 2 * d10.s3 + 1 * d11.s3 + 2 * d01.s3;

	C.sB  = -6 * d00.s0 + 6 * d10.s0 - 6 * d11.s0 + 6 * d01.s0 -
			4 * d00.s1 - 2 * d10.s1 + 2 * d11.s1 + 4 * d01.s1 -
			3 * d00.s2 + 3 * d10.s2 + 3 * d11.s2 - 3 * d01.s2 -
			2 * d00.s3 - 1 * d10.s3 - 1 * d11.s3 - 2 * d01.s3;

	C.sC =  2 * d00.s0 - 2 * d10.s0 + 1 * d00.s1 + 1 * d10.s1;
	C.sD =  2 * d00.s2 - 2 * d10.s2 + 1 * d00.s3 + 1 * d10.s3;

	C.sE  = -6 * d00.s0 + 6 * d10.s0 - 6 * d11.s0 + 6 * d01.s0 -
			3 * d00.s1 - 3 * d10.s1 + 3 * d11.s1 + 3 * d01.s1 -
			4 * d00.s2 + 4 * d10.s2 + 2 * d11.s2 - 2 * d01.s2 -
			2 * d00.s3 - 2 * d10.s3 - 1 * d11.s3 - 1 * d01.s3;

	C.sF =   4 * d00.s0 - 4 * d10.s0 + 4 * d11.s0 - 4 * d01.s0 +
			2 * d00.s1 + 2 * d10.s1 - 2 * d11.s1 - 2 * d01.s1 +
			2 * d00.s2 - 2 * d10.s2 - 2 * d11.s2 + 2 * d01.s2 +
			1 * d00.s3 + 1 * d10.s3 + 1 * d11.s3 + 1 * d01.s3;

	float ZI = 0;
	float ZXI = 0;
	float ZYI = 0;

//	if(oob)
//		alpha = 0;

	ZI += C.s0;
	ZI += C.s1 * alpha.y;
	ZI += C.s2 * alpha.y * alpha.y;
	ZI += C.s3 * alpha.y * alpha.y * alpha.y;

	ZI += C.s4 * alpha.x;
	ZI += C.s5 * alpha.y * alpha.x;
	ZI += C.s6 * alpha.y * alpha.y * alpha.x;
	ZI += C.s7 * alpha.y * alpha.y * alpha.y * alpha.x;

	ZI += C.s8 * alpha.x * alpha.x;
	ZI += C.s9 * alpha.y * alpha.x * alpha.x;
	ZI += C.sa * alpha.y * alpha.y * alpha.x * alpha.x;
	ZI += C.sb * alpha.y * alpha.y * alpha.y * alpha.x * alpha.x;

	ZI += C.sc * alpha.x * alpha.x * alpha.x;
	ZI += C.sd * alpha.y * alpha.x * alpha.x * alpha.x;
	ZI += C.se * alpha.y * alpha.y * alpha.x * alpha.x * alpha.x;
	ZI += C.sf * alpha.y * alpha.y * alpha.y * alpha.x * alpha.x * alpha.x;


	ZYI += C.s1;
	ZYI += 2*C.s2 * alpha.y;
	ZYI += 3*C.s3 * alpha.y * alpha.y;

	ZYI += C.s5 * alpha.x;
	ZYI += 2*C.s6 * alpha.y * alpha.x;
	ZYI += 3*C.s7 * alpha.y * alpha.y * alpha.x;

	ZYI += C.s9 * alpha.x * alpha.x;
	ZYI += 2*C.sa * alpha.y * alpha.x * alpha.x;
	ZYI += 3*C.sb * alpha.y * alpha.y * alpha.x * alpha.x;

	ZYI += C.sd * alpha.x * alpha.x * alpha.x;
	ZYI += 2*C.se * alpha.y * alpha.x * alpha.x * alpha.x;
	ZYI += 3*C.sf * alpha.y * alpha.y * alpha.x * alpha.x * alpha.x;


	ZXI += C.s4;
	ZXI += C.s5 * alpha.y;
	ZXI += C.s6 * alpha.y * alpha.y;
	ZXI += C.s7 * alpha.y * alpha.y * alpha.y;

	ZXI += 2*C.s8 * alpha.x;
	ZXI += 2*C.s9 * alpha.y * alpha.x;
	ZXI += 2*C.sa * alpha.y * alpha.y * alpha.x;
	ZXI += 2*C.sb * alpha.y * alpha.y * alpha.y * alpha.x;

	ZXI += 3*C.sc * alpha.x * alpha.x;
	ZXI += 3*C.sd * alpha.y * alpha.x * alpha.x;
	ZXI += 3*C.se * alpha.y * alpha.y * alpha.x * alpha.x;
	ZXI += 3*C.sf * alpha.y * alpha.y * alpha.y * alpha.x * alpha.x;


	float4 leftDeriv = read_imagef(leftDerivs, samp, writeCoord);
			//readImage(leftDerivs, convert_float2(writeCoord), 1.0f);

	//ZI = clamp(ZI, 0.0f, 1.0f);

	float b = .5f;

	float ZIt = ZI - leftDeriv.x;

	ZXI = b*ZXI+(1-b)*leftDeriv.y;
	ZYI = b*ZYI+(1-b)*leftDeriv.z;


//	if(oob){
//		ZXI = 0;
//		ZYI = 0;
//		ZIt = 0;
//		ZI = 0;
//	}

	float4 outputVal = {ZXI, ZYI, ZIt, ZI};



	if(isWithin(writeCoord, Z)){
		write_imagef(Z, writeCoord, outputVal);
	}
}
