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
include ncnn_build/benchmark/CMakeFiles/benchncnn.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/benchmark/CMakeFiles/benchncnn.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/benchmark/CMakeFiles/benchncnn.dir/flags.make

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o: ncnn_build/benchmark/CMakeFiles/benchncnn.dir/flags.make
ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o: /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/benchmark/benchncnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/benchmark && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchncnn.dir/benchncnn.cpp.o -c /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/benchmark/benchncnn.cpp

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchncnn.dir/benchncnn.cpp.i"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/benchmark/benchncnn.cpp > CMakeFiles/benchncnn.dir/benchncnn.cpp.i

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchncnn.dir/benchncnn.cpp.s"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/benchmark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/benchmark/benchncnn.cpp -o CMakeFiles/benchncnn.dir/benchncnn.cpp.s

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o.requires:

.PHONY : ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o.requires

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o.provides: ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o.requires
	$(MAKE) -f ncnn_build/benchmark/CMakeFiles/benchncnn.dir/build.make ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o.provides.build
.PHONY : ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o.provides

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o.provides.build: ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o


# Object files for target benchncnn
benchncnn_OBJECTS = \
"CMakeFiles/benchncnn.dir/benchncnn.cpp.o"

# External object files for target benchncnn
benchncnn_EXTERNAL_OBJECTS =

ncnn_build/benchmark/benchncnn: ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o
ncnn_build/benchmark/benchncnn: ncnn_build/benchmark/CMakeFiles/benchncnn.dir/build.make
ncnn_build/benchmark/benchncnn: ncnn_build/src/libncnn.a
ncnn_build/benchmark/benchncnn: /usr/lib/gcc/aarch64-linux-gnu/7/libgomp.so
ncnn_build/benchmark/benchncnn: /usr/lib/aarch64-linux-gnu/libpthread.so
ncnn_build/benchmark/benchncnn: /usr/local/lib/libvulkan.so
ncnn_build/benchmark/benchncnn: ncnn_build/glslang/glslang/libglslang.a
ncnn_build/benchmark/benchncnn: ncnn_build/glslang/SPIRV/libSPIRV.a
ncnn_build/benchmark/benchncnn: ncnn_build/glslang/glslang/libMachineIndependent.a
ncnn_build/benchmark/benchncnn: ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a
ncnn_build/benchmark/benchncnn: ncnn_build/glslang/glslang/OSDependent/Unix/libOSDependent.a
ncnn_build/benchmark/benchncnn: ncnn_build/glslang/glslang/libGenericCodeGen.a
ncnn_build/benchmark/benchncnn: ncnn_build/benchmark/CMakeFiles/benchncnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable benchncnn"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/benchmark && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchncnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/benchmark/CMakeFiles/benchncnn.dir/build: ncnn_build/benchmark/benchncnn

.PHONY : ncnn_build/benchmark/CMakeFiles/benchncnn.dir/build

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/requires: ncnn_build/benchmark/CMakeFiles/benchncnn.dir/benchncnn.cpp.o.requires

.PHONY : ncnn_build/benchmark/CMakeFiles/benchncnn.dir/requires

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/clean:
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/benchmark && $(CMAKE_COMMAND) -P CMakeFiles/benchncnn.dir/cmake_clean.cmake
.PHONY : ncnn_build/benchmark/CMakeFiles/benchncnn.dir/clean

ncnn_build/benchmark/CMakeFiles/benchncnn.dir/depend:
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/benchmark /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/benchmark /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/benchmark/CMakeFiles/benchncnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/benchmark/CMakeFiles/benchncnn.dir/depend

