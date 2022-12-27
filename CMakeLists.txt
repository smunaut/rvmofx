cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rvmofx CXX)

set(CMAKE_CXX_STANDARD 14)

# Enable warnings
if (MSVC)
	add_compile_options(/W4)
else()
	add_compile_options(-Wall)
endif()

# Deps
find_package(Torch REQUIRED)


# OpenFX
# ------

set(OPENFX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/openfx)

# Check that submodule have been initialized and updated
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/openfx/include)
  message(FATAL_ERROR
    "\n submodule(s) are missing, please update your repository:\n"
    "  > git submodule update -i\n")
endif()

set(OFX_HEADER_DIR "${OPENFX_PATH}/include")

# Support library (not used ATM)
#set(OFX_SUPPORT_HEADER_DIR "${OPENFX_PATH}/Support/include")
#set(OFX_SUPPORT_LIBRARY_DIR "${OPENFX_PATH}/Support/Library")
#
#file(GLOB SUPPORT_SOURCES
#	  "${OFX_SUPPORT_HEADER_DIR}/*.h"
#
#    "${OFX_SUPPORT_LIBRARY_DIR}/ofxsCore.cpp"
#    "${OFX_SUPPORT_LIBRARY_DIR}/ofxsImageEffect.cpp"
#    "${OFX_SUPPORT_LIBRARY_DIR}/ofxsInteract.cpp"
#    "${OFX_SUPPORT_LIBRARY_DIR}/ofxsLog.cpp"
#    "${OFX_SUPPORT_LIBRARY_DIR}/ofxsMultiThread.cpp"
#    "${OFX_SUPPORT_LIBRARY_DIR}/ofxsParams.cpp"
#    "${OFX_SUPPORT_LIBRARY_DIR}/ofxsProperty.cpp"
#    "${OFX_SUPPORT_LIBRARY_DIR}/ofxsPropertyValidation.cpp"
#    "${OFX_SUPPORT_LIBRARY_DIR}/*.h"
#)


# Target
# ------

add_library(rvmofx SHARED
	src/rvmofx.cpp
)
target_include_directories(rvmofx PRIVATE ${OFX_HEADER_DIR})
target_link_libraries(rvmofx "${TORCH_LIBRARIES}")

set_target_properties(rvmofx PROPERTIES PREFIX "")
set_target_properties(rvmofx PROPERTIES SUFFIX ".ofx")
