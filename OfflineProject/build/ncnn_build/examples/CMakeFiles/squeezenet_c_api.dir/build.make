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
include ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/flags.make

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o: ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/flags.make
ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o: /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples/squeezenet_c_api.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o -c /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples/squeezenet_c_api.cpp

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.i"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples/squeezenet_c_api.cpp > CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.i

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.s"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples/squeezenet_c_api.cpp -o CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.s

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o.requires:

.PHONY : ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o.requires

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o.provides: ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o.requires
	$(MAKE) -f ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/build.make ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o.provides.build
.PHONY : ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o.provides

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o.provides.build: ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o


# Object files for target squeezenet_c_api
squeezenet_c_api_OBJECTS = \
"CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o"

# External object files for target squeezenet_c_api
squeezenet_c_api_EXTERNAL_OBJECTS =

ncnn_build/examples/squeezenet_c_api: ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o
ncnn_build/examples/squeezenet_c_api: ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/build.make
ncnn_build/examples/squeezenet_c_api: ncnn_build/src/libncnn.a
ncnn_build/examples/squeezenet_c_api: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
ncnn_build/examples/squeezenet_c_api: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
ncnn_build/examples/squeezenet_c_api: /usr/lib/gcc/aarch64-linux-gnu/7/libgomp.so
ncnn_build/examples/squeezenet_c_api: /usr/lib/aarch64-linux-gnu/libpthread.so
ncnn_build/examples/squeezenet_c_api: /usr/lib/aarch64-linux-gnu/libvulkan.so
ncnn_build/examples/squeezenet_c_api: ncnn_build/glslang/glslang/libglslang.a
ncnn_build/examples/squeezenet_c_api: ncnn_build/glslang/SPIRV/libSPIRV.a
ncnn_build/examples/squeezenet_c_api: ncnn_build/glslang/glslang/libMachineIndependent.a
ncnn_build/examples/squeezenet_c_api: ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a
ncnn_build/examples/squeezenet_c_api: ncnn_build/glslang/glslang/OSDependent/Unix/libOSDependent.a
ncnn_build/examples/squeezenet_c_api: ncnn_build/glslang/glslang/libGenericCodeGen.a
ncnn_build/examples/squeezenet_c_api: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
ncnn_build/examples/squeezenet_c_api: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
ncnn_build/examples/squeezenet_c_api: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
ncnn_build/examples/squeezenet_c_api: ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable squeezenet_c_api"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/squeezenet_c_api.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/build: ncnn_build/examples/squeezenet_c_api

.PHONY : ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/build

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/requires: ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/squeezenet_c_api.cpp.o.requires

.PHONY : ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/requires

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/clean:
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && $(CMAKE_COMMAND) -P CMakeFiles/squeezenet_c_api.dir/cmake_clean.cmake
.PHONY : ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/clean

ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/depend:
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/examples/CMakeFiles/squeezenet_c_api.dir/depend

