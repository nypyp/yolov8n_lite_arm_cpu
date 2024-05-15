cmake_minimum_required(VERSION 3.5)
project(mqtt_publisher)

set(CMAKE_CXX_STANDARD 11)

# 寻找Paho C++库的路径
find_library(PAHO_CPP_LIB paho-mqttpp3)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# 添加可执行文件
add_executable(mqtt_publisher mqtt_publisher.cpp)

# 链接Paho C++库
target_link_libraries(mqtt_publisher ${PAHO_CPP_LIB} paho-mqtt3as)