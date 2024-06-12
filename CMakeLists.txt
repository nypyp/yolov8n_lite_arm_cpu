cmake_minimum_required(VERSION 2.8)

project(yolov8n_lite_arm_cpu)

set(TARGET yolov8n_lite_arm_cpu)

# Set ARMLINUX_ARCH_ABI to armv8 or armv7 or armv7hf
set(ARMLINUX_ARCH_ABI armv8)

# Set optimize level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
message("OPTIMIZE LEVEL: ${CMAKE_CXX_FLAGS}")
# Set 3rd module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")
set(PAHO_WITH_MQTT_C ON)
set(PAHO_BUILD_STATICb ON)
set(PAHO_BUILD_SAMPLES ON)
#set opencv lib
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add subdirectories for the external modules
add_subdirectory(external/paho-mqttpp)
add_subdirectory(external/jsoncpp)



# #find mqtt lib
# find_library(PAHO_CPP_LIB paho-mqttpp3)

# #find json lib
# find_package(JsonCpp REQUIRED)
# include_directories(${JSONCPP_INCLUDE_DIRS})

# 1. path to Paddle-Lite lib
set(LITE_DIR "${PROJECT_SOURCE_DIR}/../../../cxx")

# 2. link Paddle-Lite directory
link_directories(${LITE_DIR}/lib)

include_directories(${LITE_DIR}/include)

# 3. compile options 
if(ARMLINUX_ARCH_ABI STREQUAL "armv8")
  set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
elseif(ARMLINUX_ARCH_ABI STREQUAL "armv7")
  set(CMAKE_C_COMPILER "arm-linux-gnueabi-gcc")
  set(CMAKE_CXX_COMPILER "arm-linux-gnueabi-g++")
elseif(ARMLINUX_ARCH_ABI STREQUAL "armv7hf")
  set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
  set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
else()
  message(FATAL_ERROR "Illegal ARMLINUX_ARCH_ABI: ${ARMLINUX_ARCH_ABI}")
endif()

message(STATUS "armlinux CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "armlinux CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

add_definitions(-std=c++11 -O3 -pthread)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# 4.add executable output
add_executable(${TARGET} ${TARGET}.cc)
target_link_libraries(${TARGET} -lpaddle_light_api_shared)
target_link_libraries(${TARGET} -ldl)
target_link_libraries(${TARGET} ${OpenCV_LIBRARIES})
target_link_libraries(${TARGET} paho-mqttpp3 paho-mqtt3as)
target_link_libraries(${TARGET} jsoncpp_lib)

###############################################################
# How to use one of static libaray:                           #
#  `libpaddle_api_light_bundled.a`                            #
###############################################################
# Note: default use lite's shared library.                    #
###############################################################
# 1. Comment above line using `libpaddle_light_api_shared.so`
# 2. Undo comment below line using `libpaddle_api_light_bundled.a`

#target_link_libraries(${TARGET} ${LITE_DIR}/lib/libpaddle_api_light_bundled.a)
