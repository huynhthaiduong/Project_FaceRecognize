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
CMAKE_SOURCE_DIR = /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build

# Include any dependencies generated for this target.
include ncnn_build/examples/CMakeFiles/squeezenet.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/examples/CMakeFiles/squeezenet.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/examples/CMakeFiles/squeezenet.dir/flags.make

ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o: ncnn_build/examples/CMakeFiles/squeezenet.dir/flags.make
ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o: /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/examples/squeezenet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/squeezenet.dir/squeezenet.cpp.o -c /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/examples/squeezenet.cpp

ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/squeezenet.dir/squeezenet.cpp.i"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/examples/squeezenet.cpp > CMakeFiles/squeezenet.dir/squeezenet.cpp.i

ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/squeezenet.dir/squeezenet.cpp.s"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/examples/squeezenet.cpp -o CMakeFiles/squeezenet.dir/squeezenet.cpp.s

ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o.requires:

.PHONY : ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o.requires

ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o.provides: ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o.requires
	$(MAKE) -f ncnn_build/examples/CMakeFiles/squeezenet.dir/build.make ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o.provides.build
.PHONY : ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o.provides

ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o.provides.build: ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o


# Object files for target squeezenet
squeezenet_OBJECTS = \
"CMakeFiles/squeezenet.dir/squeezenet.cpp.o"

# External object files for target squeezenet
squeezenet_EXTERNAL_OBJECTS =

ncnn_build/examples/squeezenet: ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o
ncnn_build/examples/squeezenet: ncnn_build/examples/CMakeFiles/squeezenet.dir/build.make
ncnn_build/examples/squeezenet: ncnn_build/src/libncnn.a
ncnn_build/examples/squeezenet: /usr/local/lib/libopencv_world.so.4.3.0
ncnn_build/examples/squeezenet: /usr/lib/gcc/aarch64-linux-gnu/7/libgomp.so
ncnn_build/examples/squeezenet: /usr/lib/aarch64-linux-gnu/libpthread.so
ncnn_build/examples/squeezenet: /usr/local/lib/libvulkan.so
ncnn_build/examples/squeezenet: ncnn_build/glslang/glslang/libglslang.a
ncnn_build/examples/squeezenet: ncnn_build/glslang/SPIRV/libSPIRV.a
ncnn_build/examples/squeezenet: ncnn_build/glslang/glslang/libMachineIndependent.a
ncnn_build/examples/squeezenet: ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a
ncnn_build/examples/squeezenet: ncnn_build/glslang/glslang/OSDependent/Unix/libOSDependent.a
ncnn_build/examples/squeezenet: ncnn_build/glslang/glslang/libGenericCodeGen.a
ncnn_build/examples/squeezenet: ncnn_build/examples/CMakeFiles/squeezenet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable squeezenet"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/squeezenet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/examples/CMakeFiles/squeezenet.dir/build: ncnn_build/examples/squeezenet

.PHONY : ncnn_build/examples/CMakeFiles/squeezenet.dir/build

ncnn_build/examples/CMakeFiles/squeezenet.dir/requires: ncnn_build/examples/CMakeFiles/squeezenet.dir/squeezenet.cpp.o.requires

.PHONY : ncnn_build/examples/CMakeFiles/squeezenet.dir/requires

ncnn_build/examples/CMakeFiles/squeezenet.dir/clean:
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && $(CMAKE_COMMAND) -P CMakeFiles/squeezenet.dir/cmake_clean.cmake
.PHONY : ncnn_build/examples/CMakeFiles/squeezenet.dir/clean

ncnn_build/examples/CMakeFiles/squeezenet.dir/depend:
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/examples /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples/CMakeFiles/squeezenet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/examples/CMakeFiles/squeezenet.dir/depend

