cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME TestOpenCVCmake)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(MYSQLCONNECTORCPP_ROOT_DIR
        "${MYSQLCONNECTORCPP_ROOT_DIR}"
        CACHE
        PATH
        "Where to start looking for this component.")
find_path(MYSQLCONNECTORCPP_INCLUDE_DIR
        mysql_connection.h
        HINTS
        ${MYSQLCONNECTORCPP_ROOT_DIR}
        PATH_SUFFIXES
        include)

find_library(MYSQLCONNECTORCPP_LIBRARY
        NAMES
        mysqlcppconn
        mysqlcppconn-static
        HINTS
        ${MYSQLCONNECTORCPP_ROOT_DIR}
        PATH_SUFFIXES
        lib64
        lib)

mark_as_advanced(MYSQLCONNECTORCPP_INCLUDE_DIR MYSQLCONNECTORCPP_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MysqlConnectorCpp
        DEFAULT_MSG
        MYSQLCONNECTORCPP_INCLUDE_DIR
        MYSQLCONNECTORCPP_LIBRARY)
        
if(MYSQLCONNECTORCPP_FOUND)
    set(MYSQLCONNECTORCPP_INCLUDE_DIRS "${MYSQLCONNECTORCPP_INCLUDE_DIR}") # Add any dependencies here
    set(MYSQLCONNECTORCPP_LIBRARIES "${MYSQLCONNECTORCPP_LIBRARY}") # Add any dependencies here
    mark_as_advanced(MYSQLCONNECTORCPP_ROOT_DIR)
endif()

find_package(LAPACK)
find_package(Vulkan REQUIRED)
message(STATUS "Vulkan FOUND = ${Vulkan_FOUND}")
message(STATUS "Vulkan Include = ${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan Lib = ${Vulkan_LIBRARY}")
find_package(OpenMP REQUIRED)

#add_definitions(-DCUDA_FOUND=1)

add_subdirectory(../Dlib/dlib dlib_build)

add_subdirectory(../ncnn ncnn_build)
include_directories(../ncnn/src)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Traing src/Traing.cpp src/student.cpp src/TrainModel.cpp src/UltraFace.cpp)

target_link_libraries(Traing ${OpenCV_LIBS} ${MYSQLCONNECTORCPP_LIBRARY} dlib::dlib ncnn stdc++fs Vulkan::Vulkan)

add_executable(detect src/detect.cpp src/UltraFace.cpp src/student.cpp)

target_link_libraries(detect ${OpenCV_LIBS} ${MYSQLCONNECTORCPP_LIBRARY} dlib::dlib ncnn stdc++fs Vulkan::Vulkan)
