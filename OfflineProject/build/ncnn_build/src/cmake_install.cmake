# Install script for directory: /home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/libncnn.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/allocator.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/benchmark.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/blob.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/c_api.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/command.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/cpu.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/datareader.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/gpu.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/layer.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/layer_shader_type.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/layer_type.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/mat.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/modelbin.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/net.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/option.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/paramdict.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/pipeline.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/pipelinecache.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/simpleocv.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/simpleomp.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/simplestl.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/ncnn/src/vulkan_header_fix.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/ncnn_export.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/layer_shader_type_enum.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/layer_type_enum.h"
    "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/platform.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/kyo/Desktop/Project_FaceRecognize-master1/Project_FaceRecognize-master/OfflineProject/build/ncnn_build/src/ncnnConfig.cmake")
endif()

