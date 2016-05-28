cmake_minimum_required(VERSION 3.3)
project(comvision)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

find_package(OpenCV REQUIRED)

#set(SOURCE_FILES first_steps/first_steps.cpp)
#add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES gradients/grads.cpp)
#add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES detectors/detectors.cpp)
#add_executable(comvision ${SOURCE_FILES})

set(SOURCE_FILES adv_search.cpp)
add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES test_save.cpp)
#add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES bow_search.cpp)
#add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES test.cpp)
#add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES clusters/clustering.cpp)
#add_executable(comvision ${SOURCE_FILES})

#set(SOURCE_FILES resize.cpp)
#add_executable(comvision ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(comvision ${OpenCV_LIBS}) # replace binary_name
