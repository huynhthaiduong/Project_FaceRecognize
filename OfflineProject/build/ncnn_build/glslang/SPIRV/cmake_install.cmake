# Install script for directory: /home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/glslang/SPIRV/libSPIRV.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SPIRVTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SPIRVTargets.cmake"
         "/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/glslang/SPIRV/CMakeFiles/Export/lib/cmake/SPIRVTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SPIRVTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SPIRVTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake" TYPE FILE FILES "/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/glslang/SPIRV/CMakeFiles/Export/lib/cmake/SPIRVTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake" TYPE FILE FILES "/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/build/ncnn_build/glslang/SPIRV/CMakeFiles/Export/lib/cmake/SPIRVTargets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/glslang/SPIRV" TYPE FILE FILES
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/bitutils.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/spirv.hpp"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/GLSL.std.450.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/GLSL.ext.EXT.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/GLSL.ext.KHR.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/GlslangToSpv.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/hex_float.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/Logger.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/SpvBuilder.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/spvIR.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/doc.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/SpvTools.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/disassemble.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/GLSL.ext.AMD.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/GLSL.ext.NV.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/NonSemanticDebugPrintf.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/SPVRemapper.h"
    "/home/kyo/Desktop/KLTN/Project_FaceRecognize/ncnn/glslang/SPIRV/doc.h"
    )
endif()

