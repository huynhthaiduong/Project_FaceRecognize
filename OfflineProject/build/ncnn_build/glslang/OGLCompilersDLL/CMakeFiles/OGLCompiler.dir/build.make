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
include ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/depend.make

# Include the progress variables for this target.
include ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/progress.make

# Include the compile flags for this target's objects.
include ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/flags.make

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o: ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/flags.make
ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o: /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/glslang/OGLCompilersDLL/InitializeDll.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/glslang/OGLCompilersDLL && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o -c /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/glslang/OGLCompilersDLL/InitializeDll.cpp

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.i"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/glslang/OGLCompilersDLL && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/glslang/OGLCompilersDLL/InitializeDll.cpp > CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.i

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.s"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/glslang/OGLCompilersDLL && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/glslang/OGLCompilersDLL/InitializeDll.cpp -o CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.s

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o.requires:

.PHONY : ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o.requires

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o.provides: ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o.requires
	$(MAKE) -f ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/build.make ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o.provides.build
.PHONY : ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o.provides

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o.provides.build: ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o


# Object files for target OGLCompiler
OGLCompiler_OBJECTS = \
"CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o"

# External object files for target OGLCompiler
OGLCompiler_EXTERNAL_OBJECTS =

ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a: ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o
ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a: ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/build.make
ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a: ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libOGLCompiler.a"
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/glslang/OGLCompilersDLL && $(CMAKE_COMMAND) -P CMakeFiles/OGLCompiler.dir/cmake_clean_target.cmake
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/glslang/OGLCompilersDLL && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OGLCompiler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/build: ncnn_build/glslang/OGLCompilersDLL/libOGLCompiler.a

.PHONY : ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/build

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/requires: ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/InitializeDll.cpp.o.requires

.PHONY : ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/requires

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/clean:
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/glslang/OGLCompilersDLL && $(CMAKE_COMMAND) -P CMakeFiles/OGLCompiler.dir/cmake_clean.cmake
.PHONY : ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/clean

ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/depend:
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/glslang/OGLCompilersDLL /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/glslang/OGLCompilersDLL /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ncnn_build/glslang/OGLCompilersDLL/CMakeFiles/OGLCompiler.dir/depend
