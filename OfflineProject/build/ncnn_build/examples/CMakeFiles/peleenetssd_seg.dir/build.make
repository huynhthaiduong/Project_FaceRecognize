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
include ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/flags.make

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o: ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/flags.make
ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o: /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples/peleenetssd_seg.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o -c /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples/peleenetssd_seg.cpp

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.i"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples/peleenetssd_seg.cpp > CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.i

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.s"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples/peleenetssd_seg.cpp -o CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.s

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o.requires:

.PHONY : ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o.requires

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o.provides: ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o.requires
	$(MAKE) -f ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/build.make ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o.provides.build
.PHONY : ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o.provides

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o.provides.build: ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o


# Object files for target peleenetssd_seg
peleenetssd_seg_OBJECTS = \
"CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o"

# External object files for target peleenetssd_seg
peleenetssd_seg_EXTERNAL_OBJECTS =

ncnn_build/examples/peleenetssd_seg: ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o
ncnn_build/examples/peleenetssd_seg: ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/build.make
ncnn_build/examples/peleenetssd_seg: ncnn_build/src/libncnn.a
ncnn_build/examples/peleenetssd_seg: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
ncnn_build/examples/peleenetssd_seg: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
ncnn_build/examples/peleenetssd_seg: /usr/lib/gcc/aarch64-linux-gnu/7/libgomp.so
ncnn_build/examples/peleenetssd_seg: /usr/lib/aarch64-linux-gnu/libpthread.so
ncnn_build/examples/peleenetssd_seg: /usr/lib/aarch64-linux-gnu/libvulkan.so
ncnn_build/examples/peleenetssd_seg: ncnn_build/glslang/glslang/libglslang.a
ncnn_build/examples/peleenetssd_seg: ncnn_build/glslang/SPIRV/libSPIRV.a
ncnn_build/examples/peleenetssd_seg: ncnn_build/glslang/glslang/libMachineIndependent.a
ncnn_build/examples/peleenetssd_seg: ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a
ncnn_build/examples/peleenetssd_seg: ncnn_build/glslang/glslang/OSDependent/Unix/libOSDependent.a
ncnn_build/examples/peleenetssd_seg: ncnn_build/glslang/glslang/libGenericCodeGen.a
ncnn_build/examples/peleenetssd_seg: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
ncnn_build/examples/peleenetssd_seg: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
ncnn_build/examples/peleenetssd_seg: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
ncnn_build/examples/peleenetssd_seg: ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable peleenetssd_seg"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/peleenetssd_seg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/build: ncnn_build/examples/peleenetssd_seg

.PHONY : ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/build

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/requires: ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/peleenetssd_seg.cpp.o.requires

.PHONY : ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/requires

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/clean:
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples && $(CMAKE_COMMAND) -P CMakeFiles/peleenetssd_seg.dir/cmake_clean.cmake
.PHONY : ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/clean

ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/depend:
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/examples /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/examples/CMakeFiles/peleenetssd_seg.dir/depend
