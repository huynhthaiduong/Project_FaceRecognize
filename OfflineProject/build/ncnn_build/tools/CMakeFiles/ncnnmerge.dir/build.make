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
include ncnn_build/tools/CMakeFiles/ncnnmerge.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/tools/CMakeFiles/ncnnmerge.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/tools/CMakeFiles/ncnnmerge.dir/flags.make

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o: ncnn_build/tools/CMakeFiles/ncnnmerge.dir/flags.make
ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o: /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/ncnnmerge.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o -c /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/ncnnmerge.cpp

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.i"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/ncnnmerge.cpp > CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.i

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.s"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools/ncnnmerge.cpp -o CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.s

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o.requires:

.PHONY : ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o.requires

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o.provides: ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o.requires
	$(MAKE) -f ncnn_build/tools/CMakeFiles/ncnnmerge.dir/build.make ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o.provides.build
.PHONY : ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o.provides

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o.provides.build: ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o


# Object files for target ncnnmerge
ncnnmerge_OBJECTS = \
"CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o"

# External object files for target ncnnmerge
ncnnmerge_EXTERNAL_OBJECTS =

ncnn_build/tools/ncnnmerge: ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o
ncnn_build/tools/ncnnmerge: ncnn_build/tools/CMakeFiles/ncnnmerge.dir/build.make
ncnn_build/tools/ncnnmerge: ncnn_build/tools/CMakeFiles/ncnnmerge.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ncnnmerge"
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ncnnmerge.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/tools/CMakeFiles/ncnnmerge.dir/build: ncnn_build/tools/ncnnmerge

.PHONY : ncnn_build/tools/CMakeFiles/ncnnmerge.dir/build

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/requires: ncnn_build/tools/CMakeFiles/ncnnmerge.dir/ncnnmerge.cpp.o.requires

.PHONY : ncnn_build/tools/CMakeFiles/ncnnmerge.dir/requires

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/clean:
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools && $(CMAKE_COMMAND) -P CMakeFiles/ncnnmerge.dir/cmake_clean.cmake
.PHONY : ncnn_build/tools/CMakeFiles/ncnnmerge.dir/clean

ncnn_build/tools/CMakeFiles/ncnnmerge.dir/depend:
	cd /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/tools /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools /home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/tools/CMakeFiles/ncnnmerge.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/tools/CMakeFiles/ncnnmerge.dir/depend

