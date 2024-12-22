cmake_minimum_required(VERSION 3.26)
#　cmake -G "MinGW Makefiles" ..
# cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
# or cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
project(GaussianUtils)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE KERNELS_SOURCE_FILE 
     "src/kernels/*.cpp"
)

file(GLOB_RECURSE GPRS_SOURCE_FILE 
     "src/gprs/*.cpp"
)

file(GLOB_RECURSE MATH_SOURCE_FILE 
     "src/gpr_math/*.cpp"
)

file(GLOB_RECURSE UTILES_SOURCE_FILE 
     "src/utils/*.cpp"
)

# list(REMOVE_ITEM UTILS_SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)


# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     # 使用 OpenMP
#     message("Find OpenMP")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

find_package(Eigen3 REQUIRED)
# add_definitions(-DEIGEN_USE_MKL_ALL)

add_library(KernelsLibrary
    ${KERNELS_SOURCE_FILE} 
)

add_library(MathLibrary 
    ${MATH_SOURCE_FILE} 
)

add_library(UtilsLibrary 
    ${UTILES_SOURCE_FILE} 
)

add_library(GPRLibrary 
    ${GPRS_SOURCE_FILE} 
)

message("Eigen3_INCLUDE_DIRS: ${Eigen3_INCLUDE_DIRS}") # It is werid that this one return nothing, while eigen is founded (only in lab pc)


target_include_directories(KernelsLibrary PUBLIC include)
target_include_directories(UtilsLibrary PUBLIC include)
target_include_directories(GPRLibrary PUBLIC include)
target_include_directories(MathLibrary PUBLIC include)

# You have to use this in this pc (lab pc), while my own pc can skip this line by using 
# 
target_link_libraries (MathLibrary Eigen3::Eigen)
target_link_libraries (UtilsLibrary Eigen3::Eigen)
target_link_libraries (KernelsLibrary MathLibrary)
target_link_libraries (GPRLibrary KernelsLibrary UtilsLibrary)

add_executable(kernel_test src/kernel_test.cpp)
target_link_libraries(kernel_test GPRLibrary)

add_executable(gpr_test src/gpr_test.cpp)
target_link_libraries(gpr_test GPRLibrary)

add_executable(gpr_window_test src/gpr_window_test.cpp)
target_link_libraries(gpr_window_test GPRLibrary)