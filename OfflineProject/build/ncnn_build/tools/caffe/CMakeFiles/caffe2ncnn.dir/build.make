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
CMAKE_SOURCE_DIR = /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build

# Include any dependencies generated for this target.
include ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make

ncnn_build/tools/caffe/caffe.pb.cc: /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/caffe/caffe.proto
ncnn_build/tools/caffe/caffe.pb.cc: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running C++ protocol buffer compiler on caffe.proto"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && /usr/bin/protoc --cpp_out=/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe -I /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/caffe /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/caffe/caffe.proto

ncnn_build/tools/caffe/caffe.pb.h: ncnn_build/tools/caffe/caffe.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate ncnn_build/tools/caffe/caffe.pb.h

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make
ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/caffe/caffe2ncnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o -c /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/caffe/caffe2ncnn.cpp

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/caffe/caffe2ncnn.cpp > CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/caffe/caffe2ncnn.cpp -o CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.requires:

.PHONY : ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.requires

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.provides: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.requires
	$(MAKE) -f ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/build.make ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.provides.build
.PHONY : ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.provides

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.provides.build: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o


ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make
ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: ncnn_build/tools/caffe/caffe.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o -c /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe/caffe.pb.cc

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe/caffe.pb.cc > CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe/caffe.pb.cc -o CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.requires:

.PHONY : ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.requires

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.provides: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.requires
	$(MAKE) -f ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/build.make ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.provides.build
.PHONY : ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.provides

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.provides.build: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o


# Object files for target caffe2ncnn
caffe2ncnn_OBJECTS = \
"CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o" \
"CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o"

# External object files for target caffe2ncnn
caffe2ncnn_EXTERNAL_OBJECTS =

ncnn_build/tools/caffe/caffe2ncnn: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o
ncnn_build/tools/caffe/caffe2ncnn: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o
ncnn_build/tools/caffe/caffe2ncnn: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/build.make
ncnn_build/tools/caffe/caffe2ncnn: /usr/lib/aarch64-linux-gnu/libprotobuf.so
ncnn_build/tools/caffe/caffe2ncnn: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable caffe2ncnn"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffe2ncnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/build: ncnn_build/tools/caffe/caffe2ncnn

.PHONY : ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/build

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/requires: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.requires
ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/requires: ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.requires

.PHONY : ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/requires

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/clean:
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe && $(CMAKE_COMMAND) -P CMakeFiles/caffe2ncnn.dir/cmake_clean.cmake
.PHONY : ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/clean

ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/depend: ncnn_build/tools/caffe/caffe.pb.cc
ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/depend: ncnn_build/tools/caffe/caffe.pb.h
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/caffe /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/tools/caffe/CMakeFiles/caffe2ncnn.dir/depend

