cmake_minimum_required(VERSION 3.5)
project(helpful_vision)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

find_package(OpenCV REQUIRED)

find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
endif ()

#set(SOURCE_FILES main.cpp)
#add_executable(helpful_vision ${SOURCE_FILES})

set(SOURCE_FILES add_to_lib.cpp)
add_executable(helpful_vision ${SOURCE_FILES})

#set(SOURCE_FILES detectors.cpp)
#add_executable(helpful_vision ${SOURCE_FILES})

#set(SOURCE_FILES binary_save.cpp)
#add_executable(helpful_vision ${SOURCE_FILES})

#set(SOURCE_FILES resize.cpp)
#add_executable(helpful_vision ${SOURCE_FILES})

set (Boost_USE_STATIC_LIBS OFF) # enable dynamic linking
set (Boost_USE_MULTITHREAD ON)  # enable multithreading
find_package (Boost COMPONENTS REQUIRED filesystem)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(helpful_vision ${OpenCV_LIBS} ${Boost_LIBRARIES}) # replace binary_name
