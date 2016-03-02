package flow;

import static org.jocl.CL.*;

import org.jocl.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;




public class CLWrapper {
	
	int w = 0;
	int h = 0;
	String suffix = "";
	
	cl_platform_id platform;
	cl_context context;
	cl_device_id device;
	public cl_command_queue commandQueue;
	
	int numDevices = 0;
	
	
	public Map<String,cl_mem> memObjects;
	Map<String,cl_kernel> kernels;
	Map<String,cl_mem[]> argLists;
	
	
    public CLWrapper(){

        CL.setExceptionsEnabled(true);
        memObjects = new HashMap<String,cl_mem>();
        kernels = new HashMap<String,cl_kernel>();
        argLists = new HashMap<String,cl_mem[]>();
    }
    
   
    public void setImageSize(int w, int h){
    	this.w = w;
    	this.h = h;
    }
    
    public void setMemSuffix(String suffix){
    	this.suffix = suffix;
    }

    public int[] getImageSize(){
    	return new int[]{w, h};
    }
    
    public void cleanup(){
    	for(cl_mem m : memObjects.values())
    		clReleaseMemObject(m);
        for(cl_kernel kernel : kernels.values())
        	clReleaseKernel(kernel);
        
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }
    
    public void deleteMemObjects(){
    	for(cl_mem m : memObjects.values())
    		clReleaseMemObject(m);
    	memObjects.clear();
    }
    
    
    public String getPlatformDesc(cl_platform_id p){
    	long[] len = new long[1];
    	byte[] result = new byte[256];
    	clGetPlatformInfo(p, CL_PLATFORM_VERSION, 256,
				Pointer.to(result), len);
		String version = new String(Arrays.copyOf(result, (int)len[0]-1));

		clGetPlatformInfo(p, CL_PLATFORM_NAME, 256,
				Pointer.to(result), len);
		String name = new String(Arrays.copyOf(result, (int)len[0]-1));
		
		

		clGetPlatformInfo(p, CL_PLATFORM_VENDOR, 256,
				Pointer.to(result), len);
		String vendor = new String(Arrays.copyOf(result, (int)len[0]-1));
		
		
		return vendor + " : " + name + " : " + version;
    }
    
    public String getDeviceDesc(cl_device_id d){
    	long[] len = new long[1];
    	byte[] result = new byte[256];
    	clGetDeviceInfo(d, CL_DEVICE_NAME, 256,
				Pointer.to(result), len);
		String version = new String(Arrays.copyOf(result, (int)len[0]-1));

		
		
		return version;
    }
    
    public void printDevices(){
    	int n = 0;
    	for(cl_platform_id p : getPlatforms()){
    		int m = 0;
    		System.out.println("Platform " + n++ + ": " + getPlatformDesc(p));
    		for(cl_device_id d : getDevices(p)){
        		System.out.println("\tDevice " + m++ + ": " + getDeviceDesc(d));
    		}
    	}
    }
 
    cl_platform_id[] getPlatforms(){
    	int numPlatformsArray[] = new int[1];
    	clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
        
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        return platforms;
    }
    
    
    public cl_device_id[] getDevices(cl_platform_id p){
    	
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
  
        cl_device_id devices[] = new cl_device_id[numDevicesArray[0]];
        clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, numDevicesArray[0], devices, null);
       
        return devices;
    }
    public void setupPlatform(int platformIndex){
    	platform = getPlatforms()[platformIndex];
    }
    
    public void setupDeviceContext(int deviceIndex){
    	 // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        
        // Obtain the number of devices for the platform
        //int numDevicesArray[] = new int[1];
        //clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, null, numDevicesArray);
  
        
        // Obtain a device ID 
        //cl_device_id devices[] = new cl_device_id[numDevices];
        //clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, null);
        
        // Create a context for the selected device
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_ALL, null, null, null);
        
        
        long numBytes[] = {0};
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, null, numBytes); 

        // Obtain the cl_device_id for the first device
        numDevices = (int) numBytes[0] / Sizeof.cl_device_id;
        cl_device_id devices[] = new cl_device_id[numDevices];
        
        clGetContextInfo(context, CL_CONTEXT_DEVICES, numBytes[0], 
        		Pointer.to(devices), null);
        
        device = devices[deviceIndex];

    }
    
    public void setupCommandQueue(){
    	// Create a command-queue for the selected device
        commandQueue = clCreateCommandQueue(context, device, 0, null);
    }
    
    
    public void createImage(String name, int width, int height){
    	cl_image_format format = new cl_image_format();
    	format.image_channel_order = CL_RGBA;
    	format.image_channel_data_type = CL_UNSIGNED_INT32;
    	
	    cl_image_desc desc = new cl_image_desc();
	    	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	    	desc.image_width = width;
	    	desc.image_height = height;
	    
	    
	    memObjects.put(name, clCreateImage(context, 
	            CL_MEM_READ_WRITE,
	            format, desc,
	            null, null));
    }
    
    
    
    
    public void createImage1F(String name, int width, int height){
    	cl_image_format format = new cl_image_format();
    	format.image_channel_order = CL_LUMINANCE;
    	format.image_channel_data_type = CL_FLOAT;
    	
	    cl_image_desc desc = new cl_image_desc();
	    	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	    	desc.image_width = width;
	    	desc.image_height = height;
	    
	    
	    memObjects.put(name, clCreateImage(context, 
	            CL_MEM_READ_WRITE,
	            format, desc,
	            null, null));
    }
    
    
    public void createImage1I(String name, int width, int height){
    	cl_image_format format = new cl_image_format();
    	format.image_channel_order = CL_R;
    	format.image_channel_data_type = CL_SIGNED_INT32;
    	
	    cl_image_desc desc = new cl_image_desc();
	    	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	    	desc.image_width = width;
	    	desc.image_height = height;
	    
	    
	    memObjects.put(name, clCreateImage(context, 
	            CL_MEM_READ_WRITE,
	            format, desc,
	            null, null));
    }
    
    
    public void createBuffer1f(String name, int length){
    	createBuffer1f(name, length, CL_MEM_READ_WRITE);
    }
    
    public void createBuffer1f(String name, long length, long flags){
    	long bufferSize = (long)Sizeof.cl_float * length;
	    memObjects.put(name, clCreateBuffer(context, flags, bufferSize, null, null));
    }
    
    public void createBuffer(String name, float[] arr){
    	createBuffer(name, arr, CL_MEM_READ_WRITE);
    }
    
    public void createBuffer(String name, float[] arr, long flags){
	    memObjects.put(name, clCreateBuffer(context, flags| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * arr.length, Pointer.to(arr), null));
    }
    
    public void createBuffer(String name, int[] arr, long flags){
	    memObjects.put(name, clCreateBuffer(context, flags| CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * arr.length, Pointer.to(arr), null));
    }
    
    
    public void createBuffer(String name, Struct<Integer>[] arr, long flags){
    	int stride = arr[0].toArray().length;
    	int totalLength = arr.length * stride;
    	
    	int i = 0;
    	int[] rawData = new int[totalLength];
    	for(Struct<Integer> object : arr){
    		for(Integer element : object.toArray()){
    			rawData[i++] = element;
    		}
    	}
	    memObjects.put(name, clCreateBuffer(context, flags| CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * totalLength, Pointer.to(rawData), null));
    }
    
    
    public void createBuffer(String name, List<? extends Struct<Integer>> list, long flags){
    	
    	int stride = list.get(0).toArray().length;
    	int totalLength = list.size() * stride;
    	
    	int i = 0;
    	int[] rawData = new int[totalLength];
    	for(Struct<Integer> object : list){
    		for(Integer element : object.toArray()){
    			if(element != null)
    				rawData[i] = element;
    			i++;
    		}
    	}
	    memObjects.put(name, clCreateBuffer(context, flags| CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * totalLength, Pointer.to(rawData), null));
    }
    
    
    public <T extends Struct<Integer>>  List<Integer> createBufferFromListOfLists(String name, List<? extends List<T>> listOfLists, long flags){
    	ArrayList<Integer> retVal = new ArrayList<Integer>();
    	ArrayList<Struct<Integer>> rawList = new ArrayList<Struct<Integer>>();


    	for(List<T> list : listOfLists){
    		retVal.add(list.size());
    		rawList.addAll(list);
    	}
    	int stride = rawList.get(0).toArray().length;
    	int totalLength = rawList.size() * stride;

    	int i = 0;
    	int[] rawData = new int[totalLength];

    	for(Struct<Integer> object : rawList){
    		for(Integer element : object.toArray()){
    			if(element != null)
    				rawData[i] = element;
    			i++;

    		}
    	}
    	memObjects.put(name, clCreateBuffer(context, flags| CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * totalLength, Pointer.to(rawData), null));
    	return retVal;
    }
    
    public void createBuffer1i(String name, int length){

    	int[] arr = new int[length];
	    memObjects.put(name, clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * length, Pointer.to(arr), null));
    }
    
    public void createBuffer1i(String name, int length, long flags){

    	int[] arr = new int[length];
	    memObjects.put(name, clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * length, Pointer.to(arr), null));
    }
    
    public void createImage4F(String name, int width, int height){
        cl_image_format format = new cl_image_format();
        	format.image_channel_order = CL_RGBA;
        	format.image_channel_data_type = CL_FLOAT;
        	
        cl_image_desc desc = new cl_image_desc();
        	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        	desc.image_width = width;
        	desc.image_height = height;
        
        
        memObjects.put(name, clCreateImage(context, 
                CL_MEM_READ_WRITE,
                format, desc,
                null, null));
    }
    
    public void createImage4F(String name){
    	createImage4F(name, w, h);
    }
    
    public void createImage1F(String name){
    	createImage1F(name, w, h);
    }
    
    
    public void createImage1I(String name){
    	createImage1I(name, w, h);
    }
    
    
    public void createImageArray4F(String name, int width, int height, int numImages){
    	cl_image_format format = new cl_image_format();
    	format.image_channel_order = CL_RGBA;
    	format.image_channel_data_type = CL_FLOAT;
    	
	    cl_image_desc desc = new cl_image_desc();
	    	desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
	    	desc.image_width = width;
	    	desc.image_height = height;
	    	desc.image_array_size = numImages;
	    
	    
	    memObjects.put(name, clCreateImage(context, 
	            CL_MEM_READ_WRITE,
	            format, desc,
	            null, null));
	    

    	
        
//        for(int n = 0; n < width * height; n++){
//        	for(int colorNum = 0; colorNum < 4; colorNum++){
//        		float colorVal = 0;
//        		if(colorNum < colorDepth){
//        			colorVal = (float)pixels[n * colorDepth + colorNum];
//        		}
//        		floatPixels[4*n + colorNum] = (colorVal - offset) / mult;
//        	}
//        }
        	
        
        

    }
    
    
    public void createImage(String name, BufferedImage img){
    	int width = img.getWidth();
    	int height = img.getHeight();
    	int colorDepth = img.getColorModel().getNumComponents();

    	int[] pixels = new int[width * height * colorDepth];
        
        img.getData().getPixels(0,0,width,height,pixels);
        
        cl_image_format format = new cl_image_format();
        	format.image_channel_order = CL_RGBA;
        	format.image_channel_data_type = CL_UNSIGNED_INT32;
        	
        cl_image_desc desc = new cl_image_desc();
        	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        	desc.image_width = width;
        	desc.image_height = height;
        
        
        memObjects.put(name, clCreateImage(context, 
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                format, desc,
                Pointer.to(pixels), null));
    }
    
    
    public int[] createImage4f(String name, BufferedImage[] imgs, float mult, float offset){
    	int numImages = imgs.length;
    	int[] size = {imgs[0].getWidth(),imgs[0].getHeight()};

    	int colorDepth = 1;
    	
    	int[][] pixels = new int[numImages][];
    	for(int n = 0; n < numImages; n++){
    		pixels[n] = new int[size[0] * size[1] * colorDepth];
    		imgs[n].getData().getPixels(0,0,size[0],size[1],pixels[n]);
    	}

        float[] floatPixels = new float[size[0] * size[1] * 4];
        
        for(int n = 0; n < size[0] * size[1]; n++){
        	for(int colorNum = 0; colorNum < 4; colorNum++){
        		float colorVal = 0;
        		if(colorNum < numImages){
        			colorVal = (float)pixels[colorNum][n * colorDepth + (colorNum%colorDepth)];
        		}
        		floatPixels[4*n + colorNum] = (colorVal - offset) / mult;
        	}
        }
        	
        
        
        
        cl_image_format format = new cl_image_format();
        	format.image_channel_order = CL_RGBA;
        	format.image_channel_data_type = CL_FLOAT;
        	
        cl_image_desc desc = new cl_image_desc();
        	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        	desc.image_width = size[0];
        	desc.image_height = size[1];
        
        
        memObjects.put(name, clCreateImage(context, 
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                format, desc,
                Pointer.to(floatPixels), null));
        return size;
    }
    
    
    
    public void createImageFromSequence(String name, String seq){
    	//cl_image_format[] formats = new cl_image_format[256];
    	//clGetSupportedImageFormats(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,CL_MEM_OBJECT_IMAGE1D, 256, formats, null);
    	//for(cl_image_format f : formats)
    	//	System.out.println(f);
    	
    	int size = (seq.length());
    	
    	byte[] pixels = new byte[size];

        for(int n = 0; n < seq.length(); n++){
        	int pixel = (seq.charAt(n) - 'A');
        	//pixel += (short) (seq.charAt(n+1) - 'A') * 32;
        	//pixel += (short) (seq.charAt(n+2) - 'A') * 32 * 32;
        	
        	pixels[n] = (byte) pixel;
        }
        	
        
        
        
        cl_image_format format = new cl_image_format();
        	format.image_channel_order = CL_R;
        	format.image_channel_data_type = CL_UNSIGNED_INT8;
        	
        cl_image_desc desc = new cl_image_desc();
        	desc.image_type = CL_MEM_OBJECT_IMAGE1D;
        	desc.image_width = size;
        
        
        memObjects.put(name, clCreateImage(context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                format, desc,
                Pointer.to(pixels), null));
    }
    
    
    public int[] createImage4f(String name, BufferedImage img, float mult, float offset){
    	
    	
    	int[] size = {img.getWidth(),img.getHeight()};

    	int colorDepth = img.getColorModel().getNumComponents();
    	
    	int[] pixels = new int[size[0] * size[1] * colorDepth];
    	
    	img.getData().getPixels(0,0,size[0],size[1],pixels);
    	
    	
        float[] floatPixels = new float[size[0] * size[1] * 4];
        
        for(int n = 0; n < size[0] * size[1]; n++){
        	for(int colorNum = 0; colorNum < 4; colorNum++){
        		float colorVal = 0;
        		if(colorNum < colorDepth){
        			colorVal = (float)pixels[n * colorDepth + colorNum];
        		}
        		floatPixels[4*n + colorNum] = (colorVal - offset) / mult;
        	}
        }
        	
        
        
        
        cl_image_format format = new cl_image_format();
        	format.image_channel_order = CL_RGBA;
        	format.image_channel_data_type = CL_FLOAT;
        	
        cl_image_desc desc = new cl_image_desc();
        	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        	desc.image_width = size[0];
        	desc.image_height = size[1];
        
        
        memObjects.put(name, clCreateImage(context, 
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                format, desc,
                Pointer.to(floatPixels), null));
        return size;
    }
    
    Random r = new Random(1);
    public void createRandImage4f(String name, int width, int height){
    	float[] pixels = new float[width * height * 4];
        
        for(int i = 0; i != pixels.length; i++){
        	pixels[i] = r.nextFloat();// * 2.0f - 1;
        	//System.out.println(r.nextFloat());
        }
        cl_image_format format = new cl_image_format();
        	format.image_channel_order = CL_RGBA;
        	format.image_channel_data_type = CL_FLOAT;
        	
        cl_image_desc desc = new cl_image_desc();
        	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        	desc.image_width = width;
        	desc.image_height = height;
        
        
        memObjects.put(name, clCreateImage(context, 
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                format, desc,
                Pointer.to(pixels), null));
    }
    
    public void getImage4f(String name, BufferedImage img){
    	getImage4f(name, img, 1, 0);
    }
    
//    public void fillBuffer(String name, int length, int value){
//    	clEnqueueWriteBuffer(commandQueue, memObjects.get(name), true, 0, length*4, Pointer.to(new int[length]), 0, null, null);
//    }
    
    public float[] getBuffer1f(String name, int length){
    	float[] data = new float[length * 4];
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, 0, length*4, Pointer.to(data), 0, null, null);
    	return data;
    	
    }
    
    
    public int getInt(String name, int[] arrayPos, int stride){
    	int[] data = new int[1];
    	int offset = arrayPos[0] * stride + arrayPos[1];
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, offset * 4, 4, Pointer.to(data), 0, null, null);
    	return data[0];
    }
    
    
    public int getInt(String name, int offset){
    	int[] data = new int[1];
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, offset * 4, 4, Pointer.to(data), 0, null, null);
    	return data[0];
    }
    
    public float getFloat(String name, int offset){
    	float[] data = new float[16];
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, offset * 4, 4, Pointer.to(data), 0, null, null);
    	return data[0];
    }
    
    
    public int[][] getBuffer2i(String name, int[] arrSize, int stride){
    	int[] rawData = new int[arrSize[0] * stride];
    	int[][] data = new int[arrSize[0]][arrSize[1]];
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, 0, arrSize[0]*stride*4, Pointer.to(rawData), 0, null, null);

    	for(int row = 0; row < arrSize[0]; row++){
    		int index = row * stride;
    		for(int col = 0; col < arrSize[1]; col++){
    			data[row][col] = rawData[index++];
    		}
    		//System.out.println(Arrays.toString(data[row]));
    	}
    	return data;
    	
    }
    
    public void getBuffer(String name, float[][] data, int stride){
    	float[] rawData = new float[data.length * stride];
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, 0, data.length*stride*4, Pointer.to(rawData), 0, null, null);

    	for(int row = 0; row < data.length; row++){
    		int index = row * stride;
    		for(int col = 0; col < data[row].length; col++){
    			data[row][col] = rawData[index++];
    		}
    	}
    }
    
    public void getBuffer(String name, int[][] data, int stride){
    	int[] rawData = new int[data.length * stride];
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, 0, data.length*stride*4, Pointer.to(rawData), 0, null, null);

    	for(int row = 0; row < data.length; row++){
    		int index = row * stride;
    		for(int col = 0; col < data[row].length; col++){
    			data[row][col] = rawData[index++];
    		}
    	}
    }
    
    public void getBuffer(String name, int[] data){

    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, 0, data.length*4, Pointer.to(data), 0, null, null);
    }
    
    public void getBuffer(String name, float[] data){

    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, 0, data.length*4, Pointer.to(data), 0, null, null);
    }
    
    public void getBufferDirect(String name, int byteSize){
    	ByteBuffer buffer = ByteBuffer.allocateDirect(byteSize);
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_FALSE, 0, byteSize, Pointer.to(buffer), 0, null, null);
    }
    
    public void clearBuffer(String name, int length){
    	clEnqueueWriteBuffer(commandQueue, memObjects.get(name + suffix), false, 0, length*4, Pointer.to(new int[length]), 0, null, null);
    }
    
    public void clearBuffer(String name, boolean exactName, int length){
    	clEnqueueWriteBuffer(commandQueue, memObjects.get(name), true, 0, length*4, Pointer.to(new int[length]), 0, null, null);
    }
    
    public int[] getBuffer1i(String name, int length){
    	int[] data = new int[length];
    	clEnqueueReadBuffer(commandQueue, memObjects.get(name), CL_TRUE, 0, length*4, Pointer.to(data), 0, null, null);
    	return data;
    }
    
    public void getImage4f(String name, BufferedImage img, float mult, float offset){
    	getImage4f(name, img, new float[] {mult,mult,mult,mult}, new float[] {offset,offset,offset,offset});
    }
    
    public void getImage4f(String name, BufferedImage img, float mult[], float offset[]){
    	int width = img.getWidth();
    	int height = img.getHeight();
    	
    	float[] pixels = new float[width * height * 4];
    	
    	clEnqueueReadImage(commandQueue, memObjects.get(name), CL_TRUE, new long[] {0, 0, 0}, new long[] {width, height, 1}, 0, 0, Pointer.to(pixels), 0, null, null);
        
        for(int x = 0; x < width; x++)
        	for(int y = 0; y < height; y++){
        		int rPos = (x + y * width)*4;
            	int gPos = (x + y * width)*4+1;
            	int bPos = (x + y * width)*4+2;
            	
            	int a = 255;
            	int r = (int) (pixels[rPos] * mult[0] + offset[0]);
            	int g = (int) (pixels[gPos] * mult[1] + offset[1]);
            	int b = (int) (pixels[bPos] * mult[2] + offset[2]);
            	
            	if(r > 255) r = 255;
            	if(g > 255) g = 255;
            	if(b > 255) b = 255;
            	
            	if(r < 0) r = 0;
            	if(g < 0) g = 0;
            	if(b < 0) b = 0;
            	
            	int color = (a << 24) + (r << 16) + (g << 8) + (b << 0);
            	img.setRGB(x, y, color);	
        	}
    }
    
    
    public void getImage4f(String name, float[][][] img){
    	int width = img.length;
    	int height = img[0].length;
    	
    	float[] pixels = new float[width * height * 4];
    	clEnqueueReadImage(commandQueue, memObjects.get(name), CL_TRUE, new long[] {0, 0, 0}, new long[] {width, height, 1}, 0, 0, Pointer.to(pixels), 0, null, null);
        int index = 0;
        for(int x = 0; x < width; x++)
        	for(int y = 0; y < height; y++){
        		img[x][y][0] = pixels[index++];
        		img[x][y][1] = pixels[index++];
        		img[x][y][2] = pixels[index++];
        		img[x][y][3] = pixels[index++];
        	}
    }
    
    public void getImage2f(String name, float[][][] img){
    	int width = img.length;
    	int height = img[0].length;
    	
    	float[] pixels = new float[width * height * 4];
    	clEnqueueReadImage(commandQueue, memObjects.get(name), CL_TRUE, new long[] {0, 0, 0}, new long[] {width, height, 1}, 0, 0, Pointer.to(pixels), 0, null, null);
        int index = 0;
        for(int x = 0; x < width; x++)
        	for(int y = 0; y < height; y++){
        		img[x][y][0] = pixels[index];
        		img[x][y][1] = pixels[index+1];
        		index += 4;
        	}
    }
    
    
    public void getImage4fTop2(String name, BufferedImage img, float mult[], float offset[]){
    	int width = img.getWidth();
    	int height = img.getHeight();
    	
    	float[] pixels = new float[width * height * 4];
    	
    	clEnqueueReadImage(commandQueue, memObjects.get(name), CL_TRUE, new long[] {0, 0, 0}, new long[] {width, height, 1}, 0, 0, Pointer.to(pixels), 0, null, null);
        
        for(int x = 0; x < width; x++)
        	for(int y = 0; y < height; y++){
        		int rPos = (x + y * width)*4+2;
            	int gPos = (x + y * width)*4+3;
            	
            	
            	int a = 255;
            	int r = (int) (pixels[rPos] * mult[0] + offset[0]);
            	int g = (int) (pixels[gPos] * mult[1] + offset[1]);
            	int b = 0;
            	
            	if(r > 255) r = 255;
            	if(g > 255) g = 255;
            	if(b > 255) b = 255;
            	
            	if(r < 0) r = 0;
            	if(g < 0) g = 0;
            	if(b < 0) b = 0;
            	
            	int color = (a << 24) + (r << 16) + (g << 8) + (b << 0);
            	img.setRGB(x, y, color);	
        	}
    }
    
    public void getImage1f(String name, BufferedImage img){
    	int width = img.getWidth();
    	int height = img.getHeight();
    	
    	float[] pixels = new float[width * height];
    	
    	clEnqueueReadImage(commandQueue, memObjects.get(name), CL_TRUE, new long[] {0, 0, 0}, new long[] {width, height, 1}, 0, 0, Pointer.to(pixels), 0, null, null);
        
        for(int x = 0; x < width; x++)
        	for(int y = 0; y < height; y++){
        		int lPos = (x + y * width);
            	
            	int a = 255;
            	int r = (int) pixels[lPos];
            	int g = r;
            	int b = r;
            	
            	int color = (a << 24) + (r << 16) + (g << 8) + (b << 0);
            	img.setRGB(x, y, color);
        	}
    }
    
    public void getImage1I(String name, BufferedImage img, float mult, float offset){
    	int width = img.getWidth();
    	int height = img.getHeight();
    	
    	int[] pixels = new int[width * height];
    	
    	clEnqueueReadImage(commandQueue, memObjects.get(name), CL_TRUE, new long[] {0, 0, 0}, new long[] {width, height, 1}, 0, 0, Pointer.to(pixels), 0, null, null);
        
        for(int x = 0; x < width; x++)
        	for(int y = 0; y < height; y++){
        		int lPos = (x + y * width);
            	
            	int a = 255;
            	int r = (int) (pixels[lPos] * mult + offset);
            	int g = r;
            	int b = r;
            	
            	int color = (a << 24) + (r << 16) + (g << 8) + (b << 0);
            	img.setRGB(x, y, color);
        	}
    }
    
    public void getImage4i(String name, BufferedImage img){
    	int width = img.getWidth();
    	int height = img.getHeight();
    	
    	int[] pixels = new int[width * height * 4];
    	
    	clEnqueueReadImage(commandQueue, memObjects.get(name), CL_TRUE, new long[] {0, 0, 0}, new long[] {width, height, 1}, 0, 0, Pointer.to(pixels), 0, null, null);
        
        for(int x = 0; x < width; x++)
        	for(int y = 0; y < height; y++){
        		int rPos = (x + y * width)*4;
            	int gPos = (x + y * width)*4+1;
            	int bPos = (x + y * width)*4+2;
            	
            	int a = 255;
            	int r = (int) pixels[rPos];
            	int g = (int) pixels[gPos];
            	int b = (int) pixels[bPos];
            	
            	int color = (a << 24) + (r << 16) + (g << 8) + (b << 0);
            	img.setRGB(x, y, color);	
        	}
    }
    
    public void getImage1F(String name, BufferedImage img){
    	int width = img.getWidth();
    	int height = img.getHeight();
    	
    	float[] pixels = new float[width * height];
    	
    	clEnqueueReadImage(commandQueue, memObjects.get(name), CL_TRUE, new long[] {0, 0, 0}, new long[] {width, height, 1}, 0, 0, Pointer.to(pixels), 0, null, null);

        for(int x = 0; x < width; x++)
        	for(int y = 0; y < height; y++){
        		int L = (int)pixels[x + y * width];
            	
            	int a = 255;
            	int r = L;
            	int g = L;
            	int b = L;
            	
            	int color = (a << 24) + (r << 16) + (g << 8) + (b << 0);
            	img.setRGB(x, y, color);	
        	}
    }
    

    public void createKernelFromFile(String kernelName) throws IOException{
    	createKernelFromFile(kernelName, "");
    }
    
    public void createKernelFromFile(String kernelName, String args) throws IOException{
    	createKernelFromFile(kernelName, args, "");
    }
    
    public void createKernelFromFile(String kernelName, String args, String postfix) throws IOException{
    	File file = new File("kernels\\" + kernelName + ".cl");
    	
    	FileInputStream fis = new FileInputStream(file);
		byte[] data = new byte[(int)file.length()];
		fis.read(data);
		String programSource = new String(data, "UTF-8");
		fis.close();

		String workingDir = System.getProperty("user.dir");
		String compileArgs = args + " -Werror -I \"" + workingDir + "\\kernels\"";
		
		cl_program p = clCreateProgramWithSource(context, 1, new String[]{ programSource }, null, null);
        clBuildProgram(p, 0, null, compileArgs, null, null);
        
        kernels.put(kernelName + postfix, clCreateKernel(p, kernelName, null));
        clReleaseProgram(p);
    }

    
    public void createKernelFromFile(String kernelName, String[] memNames) throws IOException{

    	createKernelFromFile(kernelName);
    	
    	createKernelArgList(kernelName + "Args", memNames);
    	setKernelArgList(kernelName, kernelName + "Args");
    }

    public void createKernelArgList(String argListName, String[] memNames){
    	cl_mem[] argList = new cl_mem[memNames.length];
    		
    	for(int n = 0; n < memNames.length; n++){
    		argList[n] = memObjects.get(memNames[n]);
    	}
    	argLists.put(argListName, argList);
    }
    
    public void setKernelArgList(String kernelName, String argListName){
    	int argNum = 0;
    	cl_kernel kernel = kernels.get(kernelName);
    	for(cl_mem m : argLists.get(argListName)){
    		//System.out.println(m);
    		clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(m));
    	}
    }
    
    public void runKernel(String kernelName){
    	//long[] global_size = {32, 32};
    	long[] global_size = {608, 608};
    	long[] local_size = {16,16};
    	clEnqueueNDRangeKernel(commandQueue, kernels.get(kernelName), 2, null,
    			global_size, local_size, 0, null, null);
    }
    
    public void runKernel(String kernelName, long[] global_size){
    	//long[] global_size = {603, 600};
    	long[] local_size = {16,16};
    	clEnqueueNDRangeKernel(commandQueue, kernels.get(kernelName), 2, null,
    			global_size, local_size, 0, null, null);
    }
    
    
    
    
    
    private long[] getValidGlobalSize(){
    	long[] global_size = {w, h};
    	global_size[0] += 15;
    	global_size[1] += 15;
    	
    	global_size[0] /= 16;
    	global_size[1] /= 16;
    	
    	global_size[0] *= 16;
    	global_size[1] *= 16;
    	
    	return global_size;
    }
    
    public void runKernel(String kernelName, String... memNames){
    	long[] local_size = {16,16};
    	
    	
    	int argNum = 0;
    	cl_kernel kernel = kernels.get(kernelName);
    	for(String memName : memNames){
    		Pointer p;
    		if(memName.startsWith("#f")){
    			float[] in = {Float.parseFloat(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_float, p);
    		}
    		else if(memName.startsWith("#i")){
    			int[] in = {Integer.parseInt(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_int, p);
    		}
    		else{
    			p = Pointer.to(memObjects.get(memName + suffix));
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, p);
    		}
    	}
    	
    	clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
    			getValidGlobalSize(), local_size, 0, null, null);
    }
    
    
    public void setKernelArg(String kernelName, int argNum, String memName){
    	cl_kernel kernel = kernels.get(kernelName);
    	Pointer p = Pointer.to(memObjects.get(memName + suffix));
    	clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, p);
    }
    public void setKernelParams(String kernelName, String... memNames){
    	int argNum = 0;
    	cl_kernel kernel = kernels.get(kernelName);
    	for(String memName : memNames){
    		Pointer p;
    		if(memName.startsWith("#f")){
    			float[] in = {Float.parseFloat(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_float, p);
    		}
    		else if(memName.startsWith("#i")){
    			int[] in = {Integer.parseInt(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_int, p);
    		}
    		else{
    			p = Pointer.to(memObjects.get(memName + suffix));
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, p);
    		}
    	}
    }
    
    
    public cl_kernel runKernel(cl_kernel kernel, long[] global_size,long[] local_size, long[] offset){
    	clEnqueueNDRangeKernel(commandQueue, kernel, global_size.length, offset, global_size, local_size, 0, null, null);
		return kernel;
    	
    }
    public cl_kernel runKernel(String kernelName, long[] global_size,long[] local_size, long[] offset, String... memNames){
    	
    	int argNum = 0;
    	cl_kernel kernel = kernels.get(kernelName);
    	for(String memName : memNames){
    		Pointer p;
    		if(memName.startsWith("#f")){
    			float[] in = {Float.parseFloat(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_float, p);
    		}
    		else if(memName.startsWith("#i")){
    			int[] in = {Integer.parseInt(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_int, p);
    		}
    		else{
    			p = Pointer.to(memObjects.get(memName + suffix));
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, p);
    		}
    	}
    	
    	clEnqueueNDRangeKernel(commandQueue, kernel, global_size.length, offset,
    			global_size, local_size, 0, null, null);
		return kernel;
    }
    
    public void runKernel(String kernelName, boolean exactNames, String... memNames){
    	long[] local_size = {16,16};
    	
    	int argNum = 0;
    	cl_kernel kernel = kernels.get(kernelName);
    	for(String memName : memNames){
    		Pointer p;
    		if(memName.startsWith("#f")){
    			float[] in = {Float.parseFloat(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_float, p);
    		}
    		else{
    			p = Pointer.to(memObjects.get(memName));
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, p);
    		}
    		
    		
    	}
    	
    	clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
    			getValidGlobalSize(), local_size, 0, null, null);
    }
    
    public void runKernel(String kernelName, long[] global_size, String... memNames){
    	long[] local_size = {16,16};
    	
    	int argNum = 0;
    	cl_kernel kernel = kernels.get(kernelName);
    	for(String memName : memNames){
    		Pointer p;
    		if(memName.startsWith("#f")){
    			float[] in = {Float.parseFloat(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_float, p);
    		}
    		else if(memName.startsWith("#i")){
    			int[] in = {Integer.parseInt(memName.substring(2))};
    			p = Pointer.to(in);
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_int, p);
    		}
    		else{
    			p = Pointer.to(memObjects.get(memName + suffix));
    			clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, p);
    		}
    			
    		
    		
    	}
    	
    	clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
    			global_size, local_size, 0, null, null);
    }
    
    public void copyImage(String src, String dst){
    	clEnqueueCopyImage(commandQueue,  memObjects.get(src + suffix), memObjects.get(dst + suffix), new long[] {0,0,0}, new long[] {0,0,0}, new long[] {w,h,1}, 0, null, null);
    }



	public void createBufferFromString(String name, String s) {
		byte[] arr = new byte[s.length()];
		for (int i = 0; i < s.length(); i++) {
			arr[i] = (byte) (s.charAt(i) - 'A');
		}
	    memObjects.put(name, clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_char * arr.length, Pointer.to(arr), null));
	}
	
	
	public void createBufferFromMatrix(String name, int[][] matrix) {
		int[] contiguousArray = new int[matrix.length * matrix[0].length];
		int index = 0;
		for(int[] row : matrix){
			for(int val : row){
				contiguousArray[index++] = val;
			}
		}
		
	    memObjects.put(name, clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * contiguousArray.length, Pointer.to(contiguousArray), null));
	}
	
	public void createBufferFromMatrixChar(String name, int[][] matrix) {
		byte[] contiguousArray = new byte[matrix.length * matrix[0].length];
		int index = 0;
		for(int[] row : matrix){
			for(int val : row){
				contiguousArray[index++] = (byte) val;
			}
		}
		
	    memObjects.put(name, clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_char * contiguousArray.length, Pointer.to(contiguousArray), null));
	}
	
	public void createBufferFromMatrix(String name, float[][] matrix) {
		float[] contiguousArray = new float[matrix.length * matrix[0].length];
		int index = 0;
		for(float[] row : matrix){
			for(float val : row){
				contiguousArray[index++] = (byte) val;
			}
		}
		
	    memObjects.put(name, clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * contiguousArray.length, Pointer.to(contiguousArray), null));
	}
}
