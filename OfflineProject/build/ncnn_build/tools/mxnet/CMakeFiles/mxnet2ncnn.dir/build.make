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
include ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/flags.make

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o: ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/flags.make
ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o: /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/tools/mxnet/mxnet2ncnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/mxnet && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o -c /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/tools/mxnet/mxnet2ncnn.cpp

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.i"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/mxnet && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/tools/mxnet/mxnet2ncnn.cpp > CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.i

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.s"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/mxnet && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/tools/mxnet/mxnet2ncnn.cpp -o CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.s

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.requires:

.PHONY : ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.requires

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.provides: ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.requires
	$(MAKE) -f ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/build.make ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.provides.build
.PHONY : ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.provides

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.provides.build: ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o


# Object files for target mxnet2ncnn
mxnet2ncnn_OBJECTS = \
"CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o"

# External object files for target mxnet2ncnn
mxnet2ncnn_EXTERNAL_OBJECTS =

ncnn_build/tools/mxnet/mxnet2ncnn: ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o
ncnn_build/tools/mxnet/mxnet2ncnn: ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/build.make
ncnn_build/tools/mxnet/mxnet2ncnn: ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mxnet2ncnn"
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/mxnet && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mxnet2ncnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/build: ncnn_build/tools/mxnet/mxnet2ncnn

.PHONY : ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/build

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/requires: ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.requires

.PHONY : ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/requires

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/clean:
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/mxnet && $(CMAKE_COMMAND) -P CMakeFiles/mxnet2ncnn.dir/cmake_clean.cmake
.PHONY : ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/clean

ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/depend:
	cd /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/ncnn/tools/mxnet /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/mxnet /home/nghiep/Desktop/KLTN/V1.0/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/depend

