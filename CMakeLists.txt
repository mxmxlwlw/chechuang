cmake_minimum_required(VERSION 2.6)

project(main)

add_executable(main main.cpp)
find_package(opencv)
target_link_libraries(main 
    opencv_world)