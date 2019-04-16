cmake_minimum_required(VERSION 2.8)
include(ExternalProject)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(AIC)

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONs ON)

    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    find_package(Boost REQUIRED)
    find_package(Boost COMPONENTS python37 numpy37 REQUIRED)

    include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} .)
    #link_directories(${Boost_LIBRARY_DIR})

    add_library(AIC SHARED pylib.cpp)
    target_link_libraries(AIC ${PYTHON_LIBRARIES})
    #target_link_libraries(AIC ${BOOST_LIBRARIES})
    target_link_libraries(AIC Boost::python37 Boost::numpy37)

    set_target_properties(AIC PROPERTIES SUFFIX .so)
    set_target_properties(AIC PROPERTIES PREFIX "")