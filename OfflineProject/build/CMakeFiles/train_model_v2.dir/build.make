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
include CMakeFiles/train_model_v2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/train_model_v2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/train_model_v2.dir/flags.make

CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o: CMakeFiles/train_model_v2.dir/flags.make
CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o: ../src/train_model2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o -c /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/train_model2.cpp

CMakeFiles/train_model_v2.dir/src/train_model2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train_model_v2.dir/src/train_model2.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/train_model2.cpp > CMakeFiles/train_model_v2.dir/src/train_model2.cpp.i

CMakeFiles/train_model_v2.dir/src/train_model2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train_model_v2.dir/src/train_model2.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/train_model2.cpp -o CMakeFiles/train_model_v2.dir/src/train_model2.cpp.s

CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o.requires:

.PHONY : CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o.requires

CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o.provides: CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o.requires
	$(MAKE) -f CMakeFiles/train_model_v2.dir/build.make CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o.provides.build
.PHONY : CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o.provides

CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o.provides.build: CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o


CMakeFiles/train_model_v2.dir/src/student.cpp.o: CMakeFiles/train_model_v2.dir/flags.make
CMakeFiles/train_model_v2.dir/src/student.cpp.o: ../src/student.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/train_model_v2.dir/src/student.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/train_model_v2.dir/src/student.cpp.o -c /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/student.cpp

CMakeFiles/train_model_v2.dir/src/student.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train_model_v2.dir/src/student.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/student.cpp > CMakeFiles/train_model_v2.dir/src/student.cpp.i

CMakeFiles/train_model_v2.dir/src/student.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train_model_v2.dir/src/student.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/src/student.cpp -o CMakeFiles/train_model_v2.dir/src/student.cpp.s

CMakeFiles/train_model_v2.dir/src/student.cpp.o.requires:

.PHONY : CMakeFiles/train_model_v2.dir/src/student.cpp.o.requires

CMakeFiles/train_model_v2.dir/src/student.cpp.o.provides: CMakeFiles/train_model_v2.dir/src/student.cpp.o.requires
	$(MAKE) -f CMakeFiles/train_model_v2.dir/build.make CMakeFiles/train_model_v2.dir/src/student.cpp.o.provides.build
.PHONY : CMakeFiles/train_model_v2.dir/src/student.cpp.o.provides

CMakeFiles/train_model_v2.dir/src/student.cpp.o.provides.build: CMakeFiles/train_model_v2.dir/src/student.cpp.o


# Object files for target train_model_v2
train_model_v2_OBJECTS = \
"CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o" \
"CMakeFiles/train_model_v2.dir/src/student.cpp.o"

# External object files for target train_model_v2
train_model_v2_EXTERNAL_OBJECTS =

train_model_v2: CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o
train_model_v2: CMakeFiles/train_model_v2.dir/src/student.cpp.o
train_model_v2: CMakeFiles/train_model_v2.dir/build.make
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
train_model_v2: dlib_build/libdlib.a
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
train_model_v2: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
train_model_v2: /usr/local/cuda/lib64/libcudart_static.a
train_model_v2: /usr/lib/aarch64-linux-gnu/librt.so
train_model_v2: /usr/lib/aarch64-linux-gnu/libSM.so
train_model_v2: /usr/lib/aarch64-linux-gnu/libICE.so
train_model_v2: /usr/lib/aarch64-linux-gnu/libX11.so
train_model_v2: /usr/lib/aarch64-linux-gnu/libXext.so
train_model_v2: /usr/lib/aarch64-linux-gnu/libopenblas.so
train_model_v2: /usr/lib/aarch64-linux-gnu/libcublas.so
train_model_v2: /usr/lib/aarch64-linux-gnu/libcudnn.so
train_model_v2: /usr/local/cuda/lib64/libcurand.so
train_model_v2: /usr/local/cuda/lib64/libcusolver.so
train_model_v2: /usr/local/cuda/lib64/libcudart.so
train_model_v2: CMakeFiles/train_model_v2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable train_model_v2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/train_model_v2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/train_model_v2.dir/build: train_model_v2

.PHONY : CMakeFiles/train_model_v2.dir/build

CMakeFiles/train_model_v2.dir/requires: CMakeFiles/train_model_v2.dir/src/train_model2.cpp.o.requires
CMakeFiles/train_model_v2.dir/requires: CMakeFiles/train_model_v2.dir/src/student.cpp.o.requires

.PHONY : CMakeFiles/train_model_v2.dir/requires

CMakeFiles/train_model_v2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/train_model_v2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/train_model_v2.dir/clean

CMakeFiles/train_model_v2.dir/depend:
	cd /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/CMakeFiles/train_model_v2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/train_model_v2.dir/depend

