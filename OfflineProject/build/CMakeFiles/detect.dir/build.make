# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build

# Include any dependencies generated for this target.
include CMakeFiles/detect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/detect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/detect.dir/flags.make

CMakeFiles/detect.dir/src/detect.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/src/detect.cpp.o: ../src/detect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/detect.dir/src/detect.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/src/detect.cpp.o -c /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/detect.cpp

CMakeFiles/detect.dir/src/detect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/src/detect.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/detect.cpp > CMakeFiles/detect.dir/src/detect.cpp.i

CMakeFiles/detect.dir/src/detect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/src/detect.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/detect.cpp -o CMakeFiles/detect.dir/src/detect.cpp.s

CMakeFiles/detect.dir/src/detect.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/src/detect.cpp.o.requires

CMakeFiles/detect.dir/src/detect.cpp.o.provides: CMakeFiles/detect.dir/src/detect.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/src/detect.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/src/detect.cpp.o.provides

CMakeFiles/detect.dir/src/detect.cpp.o.provides.build: CMakeFiles/detect.dir/src/detect.cpp.o


CMakeFiles/detect.dir/src/UltraFace.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/src/UltraFace.cpp.o: ../src/UltraFace.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/detect.dir/src/UltraFace.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/src/UltraFace.cpp.o -c /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/UltraFace.cpp

CMakeFiles/detect.dir/src/UltraFace.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/src/UltraFace.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/UltraFace.cpp > CMakeFiles/detect.dir/src/UltraFace.cpp.i

CMakeFiles/detect.dir/src/UltraFace.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/src/UltraFace.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/UltraFace.cpp -o CMakeFiles/detect.dir/src/UltraFace.cpp.s

CMakeFiles/detect.dir/src/UltraFace.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/src/UltraFace.cpp.o.requires

CMakeFiles/detect.dir/src/UltraFace.cpp.o.provides: CMakeFiles/detect.dir/src/UltraFace.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/src/UltraFace.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/src/UltraFace.cpp.o.provides

CMakeFiles/detect.dir/src/UltraFace.cpp.o.provides.build: CMakeFiles/detect.dir/src/UltraFace.cpp.o


CMakeFiles/detect.dir/src/student.cpp.o: CMakeFiles/detect.dir/flags.make
CMakeFiles/detect.dir/src/student.cpp.o: ../src/student.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/detect.dir/src/student.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect.dir/src/student.cpp.o -c /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/student.cpp

CMakeFiles/detect.dir/src/student.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect.dir/src/student.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/student.cpp > CMakeFiles/detect.dir/src/student.cpp.i

CMakeFiles/detect.dir/src/student.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect.dir/src/student.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/student.cpp -o CMakeFiles/detect.dir/src/student.cpp.s

CMakeFiles/detect.dir/src/student.cpp.o.requires:

.PHONY : CMakeFiles/detect.dir/src/student.cpp.o.requires

CMakeFiles/detect.dir/src/student.cpp.o.provides: CMakeFiles/detect.dir/src/student.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect.dir/build.make CMakeFiles/detect.dir/src/student.cpp.o.provides.build
.PHONY : CMakeFiles/detect.dir/src/student.cpp.o.provides

CMakeFiles/detect.dir/src/student.cpp.o.provides.build: CMakeFiles/detect.dir/src/student.cpp.o


# Object files for target detect
detect_OBJECTS = \
"CMakeFiles/detect.dir/src/detect.cpp.o" \
"CMakeFiles/detect.dir/src/UltraFace.cpp.o" \
"CMakeFiles/detect.dir/src/student.cpp.o"

# External object files for target detect
detect_EXTERNAL_OBJECTS =

detect: CMakeFiles/detect.dir/src/detect.cpp.o
detect: CMakeFiles/detect.dir/src/UltraFace.cpp.o
detect: CMakeFiles/detect.dir/src/student.cpp.o
detect: CMakeFiles/detect.dir/build.make
detect: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
detect: dlib_build/libdlib.a
detect: ncnn_build/src/libncnn.a
detect: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
detect: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
detect: /usr/local/cuda/lib64/libcudart_static.a
detect: /usr/lib/aarch64-linux-gnu/librt.so
detect: /usr/lib/aarch64-linux-gnu/libSM.so
detect: /usr/lib/aarch64-linux-gnu/libICE.so
detect: /usr/lib/aarch64-linux-gnu/libX11.so
detect: /usr/lib/aarch64-linux-gnu/libXext.so
detect: /usr/lib/aarch64-linux-gnu/libopenblas.so
detect: /usr/lib/aarch64-linux-gnu/libcublas.so
detect: /usr/lib/aarch64-linux-gnu/libcudnn.so
detect: /usr/local/cuda/lib64/libcurand.so
detect: /usr/local/cuda/lib64/libcusolver.so
detect: /usr/local/cuda/lib64/libcudart.so
detect: /usr/lib/gcc/aarch64-linux-gnu/7/libgomp.so
detect: /usr/lib/aarch64-linux-gnu/libpthread.so
detect: /usr/lib/aarch64-linux-gnu/libvulkan.so
detect: ncnn_build/glslang/glslang/libglslang.a
detect: ncnn_build/glslang/SPIRV/libSPIRV.a
detect: ncnn_build/glslang/glslang/libMachineIndependent.a
detect: ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a
detect: ncnn_build/glslang/glslang/OSDependent/Unix/libOSDependent.a
detect: ncnn_build/glslang/glslang/libGenericCodeGen.a
detect: CMakeFiles/detect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable detect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/detect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/detect.dir/build: detect

.PHONY : CMakeFiles/detect.dir/build

CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/src/detect.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/src/UltraFace.cpp.o.requires
CMakeFiles/detect.dir/requires: CMakeFiles/detect.dir/src/student.cpp.o.requires

.PHONY : CMakeFiles/detect.dir/requires

CMakeFiles/detect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/detect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/detect.dir/clean

CMakeFiles/detect.dir/depend:
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles/detect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/detect.dir/depend
