# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build

# Include any dependencies generated for this target.
include ncnn_build/examples/CMakeFiles/yolov5.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/examples/CMakeFiles/yolov5.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/examples/CMakeFiles/yolov5.dir/flags.make

ncnn_build/examples/CMakeFiles/yolov5.dir/yolov5.cpp.o: ncnn_build/examples/CMakeFiles/yolov5.dir/flags.make
ncnn_build/examples/CMakeFiles/yolov5.dir/yolov5.cpp.o: /home/tamnguyen/Git_DATN/Project_FaceRecognize/ncnn/examples/yolov5.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ncnn_build/examples/CMakeFiles/yolov5.dir/yolov5.cpp.o"
	cd /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov5.dir/yolov5.cpp.o -c /home/tamnguyen/Git_DATN/Project_FaceRecognize/ncnn/examples/yolov5.cpp

ncnn_build/examples/CMakeFiles/yolov5.dir/yolov5.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5.dir/yolov5.cpp.i"
	cd /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tamnguyen/Git_DATN/Project_FaceRecognize/ncnn/examples/yolov5.cpp > CMakeFiles/yolov5.dir/yolov5.cpp.i

ncnn_build/examples/CMakeFiles/yolov5.dir/yolov5.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5.dir/yolov5.cpp.s"
	cd /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tamnguyen/Git_DATN/Project_FaceRecognize/ncnn/examples/yolov5.cpp -o CMakeFiles/yolov5.dir/yolov5.cpp.s

# Object files for target yolov5
yolov5_OBJECTS = \
"CMakeFiles/yolov5.dir/yolov5.cpp.o"

# External object files for target yolov5
yolov5_EXTERNAL_OBJECTS =

ncnn_build/examples/yolov5: ncnn_build/examples/CMakeFiles/yolov5.dir/yolov5.cpp.o
ncnn_build/examples/yolov5: ncnn_build/examples/CMakeFiles/yolov5.dir/build.make
ncnn_build/examples/yolov5: ncnn_build/src/libncnn.a
ncnn_build/examples/yolov5: /usr/local/lib/libopencv_highgui.so.4.5.2
ncnn_build/examples/yolov5: /usr/local/lib/libopencv_videoio.so.4.5.2
ncnn_build/examples/yolov5: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
ncnn_build/examples/yolov5: /usr/lib/x86_64-linux-gnu/libpthread.so
ncnn_build/examples/yolov5: /usr/local/lib/libopencv_imgcodecs.so.4.5.2
ncnn_build/examples/yolov5: /usr/local/lib/libopencv_imgproc.so.4.5.2
ncnn_build/examples/yolov5: /usr/local/lib/libopencv_core.so.4.5.2
ncnn_build/examples/yolov5: ncnn_build/examples/CMakeFiles/yolov5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable yolov5"
	cd /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/examples/CMakeFiles/yolov5.dir/build: ncnn_build/examples/yolov5

.PHONY : ncnn_build/examples/CMakeFiles/yolov5.dir/build

ncnn_build/examples/CMakeFiles/yolov5.dir/clean:
	cd /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples && $(CMAKE_COMMAND) -P CMakeFiles/yolov5.dir/cmake_clean.cmake
.PHONY : ncnn_build/examples/CMakeFiles/yolov5.dir/clean

ncnn_build/examples/CMakeFiles/yolov5.dir/depend:
	cd /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject /home/tamnguyen/Git_DATN/Project_FaceRecognize/ncnn/examples /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples /home/tamnguyen/Git_DATN/Project_FaceRecognize/OfflineProject/build/ncnn_build/examples/CMakeFiles/yolov5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/examples/CMakeFiles/yolov5.dir/depend

