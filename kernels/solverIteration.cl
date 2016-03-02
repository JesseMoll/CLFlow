#include "common.cl"

constant float sig = 1e-3f;

constant float sigQ_data = 1e-3f;
constant float sigQ_spatial = 3e-3f;

constant float  a = 0.45f;
constant float sigmaL_data = 2.5f;
constant float sigmaL_spatial = 8.0f;

typedef struct{
	float2 A;
	float2 B;
	float C;
} solverRet;

float eval(float x){
	return native_powr(sig * sig + x * x, a);
}

float deriv(float x){
	return 2 * a * x * native_powr(sig * sig + x * x, a - 1);
}


float derivOverXdata(float x){
#ifdef quadratic
	return 2 / (sigQ_data * sigQ_data);	//Quadratic
#elif charbonnier
	return 1.0f / native_sqrt(1 + x*x / sig*sig) / sig;	//Charbonnier
#elif generalized_charbonnier
	return 2 * a * native_powr(sig * sig + x * x, a - 1.0f);	//Generalized Charbonnier
#elif lorentzian
	return 2 / (2 * sigmaL_data * sigmaL_data + x * x);	//Lorentzian
#else
	return 0;
#endif
}



float derivOverXspatial(float x){
#ifdef quadratic
	return 2 / (sigQ_spatial * sigQ_spatial);	//Quadratic
#elif charbonnier
	return 1.0f / native_sqrt(1 + x*x / sig*sig) / sig;	//Charbonnier
#elif generalized_charbonnier
	return 2 * a * native_powr(sig * sig + x * x, a - 1.0f);	//Generalized Charbonnier
#elif lorentzian
	return 2 / (2 * sigmaL_data * sigmaL_spatial + x * x);	//Lorentzian
#else
	return 0;
#endif
}

float2 derivOverXspatial2(float2 v){
	float2 retVal;
	retVal.x = derivOverXspatial(v.x);
	retVal.y = derivOverXspatial(v.y);
	return retVal;
}



float2 getdduv(float2 uvSelf, float2 uv_duv[4], int n){
	float2 retVal = derivOverXspatial2(uvSelf - uv_duv[n]);
	return retVal;
}

solverRet getCoefficients(float4 uv_duvSelf, float2 uv[4], float3 I, image2d_t im, float lambda){

	float2 uvSelf = uv_duvSelf.xy + uv_duvSelf.zw;
	int2 gid = {get_global_id(0), get_global_id(1)};


	float3 I_2 = {I.x * I.x, I.y * I.y, I.x * I.y};
	float2 Itxy = I.xy * I.z;

	I.z += dot(I.xy, uv_duvSelf.zw);  //Linearize It

	float pp_d = derivOverXdata(I.z);

	solverRet retVal;
	retVal.B = 0;
	float2 SelfA = 0;

#pragma unroll
	for(int n = 0; n < 4; n++){
		float2 tmp = -getdduv(uvSelf, uv,n);
		SelfA -= tmp;
		retVal.B -= tmp * uv[n];
	}

	retVal.A = SelfA;
	retVal.B -= SelfA * (uv_duvSelf.xy);

	retVal.B /= lambda;
	retVal.A /= lambda;

	retVal.A += pp_d * I_2.xy;
	retVal.B -= pp_d * Itxy;
	retVal.C  = pp_d * I_2.z;

	//retVal.B  -= uv_duvSelf.wz / (pp_d * I_2.z);

	return retVal;
}


__kernel void solverIteration(
		float lambda,

		__read_only image2d_t partialDerivs, //{Ix,Iy,It, I}


		__read_only image2d_t leftDeriv,
		__read_only image2d_t rightDeriv,

		__read_only image2d_t previous,//u, v, du, dv
		__write_only image2d_t output)
{
	int2 gid = {get_global_id(0), get_global_id(1)};


	float2 uv[4];
	float4 uv_duvSelf;

	int2 Adder[] = {LEFT, RIGHT, UP, DOWN};

	//get duv and uv for the current pixel
	uv_duvSelf = read_imagef(previous, sampExact, gid);

	//read the neighborhood uv values
#pragma unroll
	for(int n = 0; n < 4; n++){
		float4 uv_duv = read_imagef(previous, samp, gid + Adder[n]);
		uv[n] = uv_duv.xy + uv_duv.zw;
	}

	//Get the partial derivatives for the warped image
	float3 I = read_imagef(partialDerivs, sampExact, gid).xyz;

	//Get the local linear equation (spatial weight increases each iteration)
	solverRet eq = getCoefficients(uv_duvSelf, uv, I, previous, lambda);

	//Solution to the local linear equation [u,v] * A = B + C * [v,u]
	//float2 outputVal = (eq.B * eq.A.yx - eq.C * eq.B.yx) / (eq.A.xy * eq.A.yx - eq.C * eq.C);
	//float2 outputVal = (eq.B * eq.A.yx - eq.C * eq.B.yx) / (eq.A.xy * eq.A.yx - eq.C * eq.C + .1f);

	float2 outputVal = (eq.B) / (eq.A);


	//clamp the result (we are only looking for subpixel motion)
	outputVal = clamp(outputVal, -1.75f, 1.75f);

	float alpha = 1.0f;
	uv_duvSelf.zw = outputVal * alpha + uv_duvSelf.zw * (1 - alpha);
	if(isWithin(gid, output))
		write_imagef(output, gid, uv_duvSelf);
}
