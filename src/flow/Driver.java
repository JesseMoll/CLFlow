package flow;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

public class Driver {

	private static CLWrapper w = new CLWrapper();

	final static Map<String, Double> pyramid_ratios = new HashMap<String, Double>();

	final static int level_iters = 3;
	final static int linear_iters = 1;
	final static int solver_iters = 15;
	final static int gncIters = 3;

	private static int getNumPyramidLevels(String prefix) {
		double temp = Math.log(Math.min(imSize[0], imSize[1]) / 16.0f);
		temp /= Math.log(pyramid_ratios.get(prefix));
		return (int) (temp + 1);
	}

	public static void createKernels() throws IOException {
		w.createKernelFromFile("resize");
		w.createKernelFromFile("scaleUV");
		w.createKernelFromFile("getImageDerivs");
		w.createKernelFromFile("warpImage");
		w.createKernelFromFile("transformImage");

		w.createKernelFromFile("weightedMedian");

		// Create a solver kernel for each of the penalty functions
		//w.createKernelFromFile("solverIteration", "-D charbonnier", "_c");
		w.createKernelFromFile("solverIteration", "-D generalized_charbonnier", "_gc");
		w.createKernelFromFile("solverIteration", "-D quadratic", "_quad");
		//w.createKernelFromFile("solverIteration", "-D lorentzian", "_l");

		w.createKernelFromFile("detectOcclusion");
		w.createKernelFromFile("clearCount");
		w.createKernelFromFile("clearImage");
		w.createKernelFromFile("copyImage");
		w.createKernelFromFile("rgb2LAB");
		w.createKernelFromFile("getMax");

		//w.createKernelFromFile("getEdges");
		//w.createKernelFromFile("dilateEdges");
		w.createKernelFromFile("compareImages");
	}

	public static float[][][] calculateFlow(String fileName, boolean writeImage, int writeSequence)
			throws IOException {

		// Read the images
		readImages(fileName);

		// Set the pyramid ratios
		pyramid_ratios.put("pre", 2.0);
		pyramid_ratios.put("gnc", 1.125);

		allocateGPUMem("pre");
		allocateGPUMem("gnc", gncIters);

		constructPyramid("pre");
		constructPyramid("gnc", gncIters);

		calculateFlow("pre", "_quad");
		calculateFlow("gnc", "_gc", gncIters);

		// read the output from the GPU
		float[][][] uv = new float[imSize[0]][imSize[1]][2];
		w.getImage2f("uvOut", uv);

		if (writeImage)
			outputFlowImage();

		if (writeSequence > 1)
			outputImageSequence(writeSequence);

		w.deleteMemObjects();
		return uv;
	}

	public static String[] getAllImageDirectories() {
		File file = new File("Images");
		String[] directories = file.list(new FilenameFilter() {
			@Override
			public boolean accept(File current, String name) {
				return new File(current, name).isDirectory();
			}
		});
		return directories;
	}

	public static void main(String[] args) throws IOException {

		//args = "--output-flow --device-num 1 --calculate-error --output-flow-image --all".split(" ");
		
		int deviceNum = 0;
		int platformNum = 0;

		boolean outputFlowImage = false;
		boolean outputFlowArray = false;
		int outputSequence = 0;
		boolean calculateError = false;
		boolean processAllImages = false;

		List<String> folderNames = new ArrayList<String>();
		for(int n = 0; n < args.length; n++){
			String arg = args[n];
			if(arg.startsWith("--")){
				switch(arg){
				case "--cl-info":
					w.printDevices();
					System.exit(0);
				case "--calculate-error":
					calculateError = true;
					break;
				case "--device-num":
					deviceNum = Integer.parseInt(args[++n]);
					break;
				case "--platform-num":
					platformNum = Integer.parseInt(args[++n]);
					break;
				case "--output-flow":
					outputFlowArray = true;
					break;
				case "--output-flow-image":
					outputFlowImage = true;
					break;
				case "--output-sequence":
					outputSequence = Integer.parseInt(args[++n]);
					break;
				case "--all":
					processAllImages = true;
					break;
				default:
					System.err.println("Invalid Arguments");
					System.exit(1);
				}
			}
			else {
				folderNames.add(arg);
			}
		}



		if(processAllImages != folderNames.isEmpty()){
			System.out.flush();
			System.err.println("Invalid Arguments");
			System.exit(1);
		}

		if(processAllImages)
			folderNames = Arrays.asList(getAllImageDirectories());

		// The OpenCL Platform (e.g. AMD, Intel, etc...)
		w.setupPlatform(platformNum);
		// The Device Number (e.g. if there are multiple video cards or
		// processors in the platform)
		w.setupDeviceContext(deviceNum);
		w.setupCommandQueue();

		// Load and compile the OpenCL Kernels
		// TODO kernel compilation caching
		createKernels();

		for (String fileName : folderNames) 
		{
			long t = System.nanoTime();
			System.out.println("Calculating flow for " + fileName);
			float[][][] uv = calculateFlow("Images/" + fileName, outputFlowImage, outputSequence);
			System.out.println("Done, elapsed time = " + (System.nanoTime() - t) / 1e6 + "ms");

			if(outputFlowArray)
				writeFlow(uv, "uv.flo");
			if (calculateError) {
				// Calculate the AEPE if the ground truth exists
				try {
					float[][][] uvGT = readGT("flow10.flo");
					System.out.println("AEPE = " + getEPE(uv, uvGT));
				} catch (IOException e) {
					// Its ok, we just don't have a ground truth
				}
			}

		}
		w.cleanup();
	}

	static double getEPE(float[][][] uv, float[][][] uvGT) {
		double EPE = 0;
		//double ang = 0;
		int total = 0;

		int h = uv[0].length;
		int w = uv.length;
		for (int x = 0; x < w; x++) {
			for (int y = 0; y < h; y++) {
				if (Math.max(uvGT[x][y][0], uvGT[x][y][1]) > 1e9)
					continue;

				//Calculate the average angular error
				//double n = 1.0 / Math.sqrt(uv[x][y][0] * uv[x][y][0] + uv[x][y][1] * uv[x][y][1] + 1);
				//double un = n * uv[x][y][0];
				//double vn = n * uv[x][y][1];

				//double tn = 1.0 / Math.sqrt(uvGT[x][y][0] * uvGT[x][y][0] + uvGT[x][y][1] * uvGT[x][y][1] + 1);
				//double tun = tn * uvGT[x][y][0];
				//double tvn = tn * uvGT[x][y][1];

				//double angErr = Math.acos(un * tun + vn * tvn + (n * tn)) * 180 / Math.PI;
				//if (Double.isFinite(angErr) && tun != 0.0 && tvn != 0.0)
				//	ang += angErr;

				double xDiff = (uvGT[x][y][0] - uv[x][y][0]);
				double yDiff = (uvGT[x][y][1] - uv[x][y][1]);

				EPE += Math.sqrt(xDiff * xDiff + yDiff * yDiff);

				total += 1;
			}
		}
		// System.out.println("Avg. angular error = " + ang / total);
		return EPE / total;
	}

	static ByteBuffer fillBuff4(DataInputStream ds, ByteBuffer bb) throws IOException {
		bb.clear();
		bb.put(ds.readByte()).put(ds.readByte()).put(ds.readByte()).put(ds.readByte()).flip();
		return bb;
	}

	static void outputBuff4(DataOutputStream ds, ByteBuffer bb) throws IOException {
		bb.flip();
		for(int n = 0; n < 4; n++)
			ds.writeByte(bb.get());
		bb.clear();
	}

	static float[][][] readGT(String filename) throws IOException {

		DataInputStream ds = new DataInputStream(
				new BufferedInputStream(new FileInputStream(new File(folderName + "\\" + filename))));
		ByteBuffer bb = ByteBuffer.allocateDirect(16);
		bb.order(ByteOrder.LITTLE_ENDIAN);

		if(fillBuff4(ds, bb).getFloat() != 202021.25f){
			throw new IOException("Invalid flo file");
		}

		int w = fillBuff4(ds, bb).getInt();
		int h = fillBuff4(ds, bb).getInt();

		float[][][] retVal = new float[w][h][2];

		for (int x = 0; x < w; x++) {
			for (int y = 0; y < h; y++) {
				retVal[x][y][0] = fillBuff4(ds, bb).getFloat();
				retVal[x][y][1] = fillBuff4(ds, bb).getFloat();
			}
		}

		return retVal;
	}

	static void writeInt(DataOutputStream ds, ByteBuffer bb, int value) throws IOException{
		bb.clear();
		bb.putInt(value);
		outputBuff4(ds, bb);
	}

	static void writeFloat(DataOutputStream ds, ByteBuffer bb, float value) throws IOException{
		bb.clear();
		bb.putFloat(value);
		outputBuff4(ds, bb);
	}

	static void writeFlow(float[][][] flow, String filename) throws IOException {

		DataOutputStream ds = new DataOutputStream(
				new BufferedOutputStream(new FileOutputStream(new File(folderName + "\\" + filename))));
		ByteBuffer bb = ByteBuffer.allocateDirect(16);
		bb.order(ByteOrder.LITTLE_ENDIAN);

		ds.write("PIEH".getBytes());


		int w = flow.length;
		int h = flow[0].length;

		writeInt(ds, bb, w);
		writeInt(ds, bb, h);

		for (int x = 0; x < w; x++) {
			for (int y = 0; y < h; y++) {
				writeFloat(ds, bb, flow[x][y][0]);
				writeFloat(ds, bb, flow[x][y][1]);
			}
		}
		ds.close();
	}

	static void calculateFlow(String prefix, String penalty) throws IOException {
		calculateFlow(prefix, penalty, getNumPyramidLevels(prefix));
	}

	static void calculateFlow(String prefix, String penalty, int levels) throws IOException {
		for (int n = levels - 1; n >= 0; n--) {
			for (int i = 0; i < (level_iters); i++) {
				compute_flow_base(n, prefix, penalty, 7, 7);
			}
		}
		w.runKernel("scaleUV", true, "uv_duvB" + prefix + "0", "uvOut");
	}

	static void outputFlowImage() throws IOException {
		w.setMemSuffix("");

		w.runKernel("clearCount", new long[] { 16, 16 }, "globalBuff");
		w.runKernel("getMax", true, "uvOut", "globalBuff");
		int[] maxUV = w.getBuffer1i("globalBuff", 2);

		UVMult = 255.0f / (Math.max(maxUV[0] / 1000.0f, maxUV[1] / 1000.0f) * 2);
		BufferedImage imgOut = new BufferedImage(imSize[0], imSize[1], BufferedImage.TYPE_INT_RGB);
		w.getImage4f("uvOut", imgOut, new float[] { UVMult, UVMult, 0 }, new float[] { 127.5f, 127.5f, 0 });
		ImageIO.write(imgOut, "png", new File(folderName + "\\UV.png"));

		// w.runKernel("transformImage", true, "left0", "uvOut", "#f1",
		// "warped");
		// w.getImage4f("warped", imgOut, 1, 0);
		// ImageIO.write(imgOut, "png", new File(folderName + "\\Frame21.png"));
	}

	static void outputImageSequence(int numFrames) throws IOException {
		numFrames--;
		File theDir = new File(folderName + "\\output");

		// if the directory does not exist, create it
		if (!theDir.exists()) {
			try {
				theDir.mkdir();
			} catch (SecurityException se) {
				System.out.println("error creating the dir");
			}
		}
		BufferedImage imgOut = new BufferedImage(imSize[0], imSize[1], BufferedImage.TYPE_INT_RGB);
		int imNum = 0;
		for (int n = 0; n <= numFrames; n++) {
			float ratio = (float) n / numFrames;
			w.runKernel("transformImage", true, "left0", "uvOut", "#f" + ratio, "warped");
			w.getImage4f("warped", imgOut, 1, 0);
			ImageIO.write(imgOut, "jpg", new File(folderName + "\\output\\warped" + (imNum++) + ".jpg"));
		}
	}

	static String folderName;
	static int[] imSize = { 603, 600 };

	static int imgNum = 1;
	static float UVMult = 255.0f / 43.2434f;

	public static void readImages(String folder) throws IOException {
		folderName = folder;

		File f1 = new File(folderName + "\\frame10.png");
		if (!f1.exists())
			f1 = new File(folderName + "\\frame10.jpg");
		File f2 = new File(folderName + "\\frame11.png");
		if (!f2.exists())
			f2 = new File(folderName + "\\frame11.jpg");

		imSize = w.createImage4f("left0", ImageIO.read(f1), 1, 0);
		w.createImage4f("right0", ImageIO.read(f2), 1, 0);
	}

	public static void constructPyramid(String prefix) throws IOException {
		constructPyramid(prefix, getNumPyramidLevels(prefix));
	}

	public static void constructPyramid(String prefix, int num_pyramid_levels) throws IOException {
		double pyramid_ratio = pyramid_ratios.get(prefix);
		w.setImageSize((int) (imSize[0]), (int) (imSize[1]));
		w.setMemSuffix("");

		w.runKernel("rgb2LAB", "left0", "right0", "LAB" + prefix + "0");

		for (int n = 0; n < num_pyramid_levels; n++) {
			double div = Math.pow(pyramid_ratio, -n);

			w.setImageSize((int) (imSize[0] * div), (int) (imSize[1] * div));
			w.setMemSuffix(prefix + n);

			if (n != 0)
				w.runKernel("resize", true, "LAB" + prefix + (n - 1), "LAB" + prefix + n);

			w.runKernel("getImageDerivs", "LAB", "leftDerivs", "rightDerivs");
			w.runKernel("clearImage", "uv_duvA");

		}
	}

	public static void allocateGPUMem(String prefix) throws IOException {
		allocateGPUMem(prefix, getNumPyramidLevels(prefix));
	}

	public static void allocateGPUMem(String prefix, int num_pyramid_levels) throws IOException {
		double pyramid_ratio = pyramid_ratios.get(prefix);
		w.createBuffer1i("globalBuff", 16);

		w.setImageSize((int) (imSize[0]), (int) (imSize[1]));
		w.createImage4F("uvOut");
		w.createImage4F("warped");

		for (int i = 0; i < num_pyramid_levels; i++) {
			String n = prefix + i;
			double div = Math.pow(pyramid_ratio, -i);
			w.setImageSize((int) (imSize[0] * div), (int) (imSize[1] * div));
			// w.createImage1I("Sobel" + n);
			w.createImage1I("Edges" + n);
			w.createImage4F("LAB" + n);
			w.createImage4F("rightDerivs" + n);
			w.createImage4F("leftDerivs" + n);
			w.createImage4F("uv_duvA" + n);
			w.createImage4F("uv_duvB" + n);
			w.createImage4F("Itxy" + n);
			w.createImage4F("occlusion" + n);
			w.createImage4F("warped" + n);
		}
	}

	private static double iterFunc(double alpha, float start, float end) {
		return start * (1 - alpha) + end * alpha;
	}

	public static void compute_flow_base(int level, String kernelPostfix) throws IOException {
		compute_flow_base(level, "", "_l", 1, 1);
	}

	private static int getIterCount(double div, String type) {
		return solver_iters;
	}

	static String lastUVprefix = "";

	public static void compute_flow_base(int level, String prefix, String kernelPostfix, float start, float end)
			throws IOException {

		double pyramid_ratio = pyramid_ratios.get(prefix);
		float div = (float) Math.pow(pyramid_ratio, -level);

		int solver_iters = getIterCount(div, kernelPostfix);

		w.setImageSize((int) (imSize[0] * div), (int) (imSize[1] * div));
		w.setMemSuffix(prefix + level);

		if (!lastUVprefix.equals("")) {
			w.runKernel("scaleUV", true, "uv_duvB" + lastUVprefix, "uv_duvA" + prefix + level);
		}
		lastUVprefix = prefix + level;

		w.runKernel("warpImage", "leftDerivs", "rightDerivs", "LAB", "uv_duvA", "Itxy");

		int iterCount = 0;

		for (int i = 0; i < linear_iters; i++) {

			for (int n = 0; n < solver_iters / 2; n++) {
				w.runKernel("solverIteration" + kernelPostfix,
						"#f" + iterFunc(iterCount++ / (double) solver_iters, start, end), "Itxy", "leftDerivs",
						"rightDerivs", "uv_duvA", "uv_duvB");
				w.runKernel("solverIteration" + kernelPostfix,
						"#f" + iterFunc(iterCount++ / (double) solver_iters, start, end), "Itxy", "leftDerivs",
						"rightDerivs", "uv_duvB", "uv_duvA");
			}

			if (solver_iters % 2 == 1) {
				w.runKernel("solverIteration" + kernelPostfix,
						"#f" + iterFunc(iterCount++ / (double) solver_iters, start, end), "Itxy", "leftDerivs",
						"rightDerivs", "uv_duvA", "uv_duvB");
				w.runKernel("copyImage", "uv_duvB", "uv_duvA");
			}
			// median filtering only around edges... unfinished
			// w.runKernel("getEdges", "uv_duvA", "Sobel");
			// w.runKernel("dilateEdges", "Sobel", "Edges");

			w.runKernel("warpImage", "leftDerivs", "rightDerivs", "LAB", "uv_duvA", "Itxy");
			w.runKernel("detectOcclusion", "uv_duvA", "LAB", "Itxy", "occlusion");

			w.runKernel("weightedMedian", "uv_duvB", "occlusion", "uv_duvA");
			w.runKernel("copyImage", "uv_duvA", "uv_duvB");
		}
	}
}
