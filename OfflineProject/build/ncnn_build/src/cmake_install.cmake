# Install script for directory: /home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/libncnn.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/allocator.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/benchmark.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/blob.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/c_api.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/command.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/cpu.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/datareader.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/gpu.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/layer.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/layer_shader_type.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/layer_type.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/mat.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/modelbin.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/net.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/option.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/paramdict.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/pipeline.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/pipelinecache.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/simpleocv.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/simpleomp.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/simplestl.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/ncnn/src/vulkan_header_fix.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/ncnn_export.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/layer_shader_type_enum.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/layer_type_enum.h"
    "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/platform.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/phamlong0906/Desktop/Project/Project_FaceRecognize/OfflineProject/build/ncnn_build/src/ncnnConfig.cmake")
endif()

