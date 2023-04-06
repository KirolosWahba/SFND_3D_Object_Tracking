#########################################################################################
# NOTE: Make sure to set all bVis flags to false in ./src/FinalProject_Camera.cpp ...	#
#  		and then build the project again for this script to run properly.				#
#########################################################################################

import os
import subprocess

if not os.path.exists("report"):
	os.makedirs("report")

os.chdir("./build")

detectors = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
descriptors = ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]

cnt = 1

for detector in detectors:
	for descriptor in descriptors:
		matchingDescriptorType = "DES_BINARY" if descriptor != "SIFT" else "DES_HOG"
		
		state, output_string = subprocess.getstatusoutput('./3D_object_tracking %s %s %s' % (detector, descriptor, matchingDescriptorType))
		final_string = (str(cnt) + " (" + detector + "/" + descriptor + "):\n" + 20*"=" + "\n\n" + output_string + "\n")

		with open("../report/log_"+ str(cnt) + "_" + detector + "_" + descriptor + ".txt", "w") as f:
			f.write(final_string)
		
		final_string_list = []
		cnt += 1
