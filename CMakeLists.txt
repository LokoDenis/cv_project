cmake_minimum_required(VERSION 3.3)
project(comvision)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

#set(SOURCE_FILES first-steps.cpp)
#add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES grads_and_etc.cpp)
#add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES detectors.cpp)
#add_executable(comvision ${SOURCE_FILES})

set(SOURCE_FILES base_creation.cpp)
add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES clustering.cpp)
#add_executable(comvision ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(comvision ${OpenCV_LIBS}) # replace binary_name
