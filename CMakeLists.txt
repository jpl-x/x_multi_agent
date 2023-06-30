cmake_minimum_required(VERSION 2.18...3.16)

set(VERSION 1.2.4)
project(x VERSION ${VERSION} LANGUAGES C CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

set(CMAKE_INSTALL_PREFIX /opt/x)

# Set build flags, depending on the architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

#################################################################################
# User build settings

# Set ON to process image and inertial data on different
# threads
option(MULTI_THREAD "Multi thread" ON)

# Set false to disable all publishing and standard output
# stream, except pose at update rate. That will improve runtime.
option(VERBOSE "Publish std out and other data" OFF)

# Set ON to enable timers
option(TIMING "Publish timing information" OFF)

# Set ON to disable compiler flags which are not
# compatible with Callgrind profiling tool.
option(PROFILING "Enable profiling flags" OFF)

# Set to ON to enable photometric calibration as in https://arxiv.org/abs/2012.14292
option(PHOTOMETRIC_CALI "Enable photometric calibration for thermal images" OFF)

# Set to ON to enable Multi UAV system
option(MULTI_UAV "Enable multi-UAV collaboration" OFF)

# Set to ON if you want to enable the request protocol for the communication
option(REQUEST_COMM "Enable request-response communication pipeline" OFF)

# Set to ON for using the ground truth matches for multi UAV
option(GT_DEBUG "Expect landmarks position ground truth as input" OFF)

#################################################################################

# Set definitions
if (GT_DEBUG)
    if (PHOTOMETRIC_CALI)
        message(WARNING "PHOTOMETRIC CALIBRATION DOES NOT SUPPORT FEATURES GT.")
        message(WARNING "PHOTOMETRIC_CALI SET TO FALSE.")
        set(PHOTOMETRIC_CALI false)
    endif ()
    if (REQUEST_COMM)
        message(WARNING "REQUEST COMM DOES NOT SUPPORT FEATURES GT.")
        message(WARNING "REQUEST_COMM SET TO FALSE.")
        set(REQUEST_COMM false)
    endif ()
    add_definitions(-DGT_DEBUG)
    message(STATUS "GT DEBUG: ON")
endif ()

if (MULTI_THREAD)
    message(STATUS "Multi thread: ON")
    add_definitions(-DMULTI_THREAD)
endif ()

if (VERBOSE)
    message(STATUS "Verbose: ON")
    add_definitions(-DVERBOSE)
endif ()

if (TIMING)
    message(STATUS "Timing: ON")
    add_definitions(-DTIMING)
endif ()

if (PHOTOMETRIC_CALI)
    add_definitions(-DPHOTOMETRIC_CALI)
    message(STATUS "Photometric calibration: ON")
endif ()

if (MULTI_UAV)
    add_definitions(-DMULTI_UAV)
    message(STATUS "Multi UAV support: ON")
    if (REQUEST_COMM)
        add_definitions(-DREQUEST_COMM)
        message(STATUS "REQUEST_COMM: ON")
    endif ()
endif ()

add_definitions(-D_LINUX -D_REENTRANT)


set(CMAKE_INSTALL_LIBDIR lib)
#################################
#
#          OpenCV >= 3.3.1
#
#################################
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    if (OpenCV_VERSION VERSION_LESS "3.3.1")
        message(FATAL_ERROR "OpenCV >= 3.3.1 not found")
    else ()
        message(STATUS "OpenCV ${OpenCV_VERSION} found.")
    endif ()
else ()
    message(FATAL_ERROR "OpenCV >= 3.3.1 not found")
endif ()

#################################
#
#             Boost
#
#################################
find_package(Boost 1.71.0 COMPONENTS log thread system filesystem REQUIRED)
message(STATUS "Boost ${Boost_VERSION} found.")

# Set libraries for multi UAV setup
if (MULTI_UAV)
    #################################
    #
    #             DBoW3
    #
    #################################
    add_subdirectory(third_party/DBow3)
    set(CONFIG_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/third_party/include")

    #################################
    #
    #              NLopt
    #
    #################################
    find_package(NLopt 2.7.1 REQUIRED)
    if (NLopt_FOUND)
        message(STATUS "NLopt ${NLopt_VERSION} found.")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${NLOPT_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NLOPT_CXX_FLAGS}")
    endif ()

    # store all libs and dirs in local vars
    set(MULTI_UAV_DIRS ${NLOPT_INCLUDE_DIRS} third_party/DBow3/include)
    set(MULTI_UAV_LIBS ${NLOPT_LIBRARIES} DBow3)
endif ()

# Set libraries photometric calibration setup
if (PHOTOMETRIC_CALI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mfpmath=sse")

    #################################
    #
    #             Ceres
    #
    #################################
    find_package(Ceres 2.0.0 REQUIRED)
    if (CERES_FOUND)
        message(STATUS "Ceres ${CERES_VERSION} found.")
    endif ()

    #################################
    #
    #             OpenMP
    #
    #################################
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(OMP_NUM_THREADS 4)
        message(STATUS "OpenMP ${OpenMP_VERSION} found")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif ()

    set(PHOTOMETRIC_DIRS ${Ceres_INCLUDE_DIR} ${OpenMP_INCLUDE_DIRS})
    set(PHOTOMETRIC_LIBS ${CERES_LIBRARIES} ${OpenMP_LIBRARIES})
endif ()

# Eigen plugin
add_definitions(-DEIGEN_MATRIXBASE_PLUGIN=<x/common/eigen_matrix_base_plugin.h>)

find_package(Eigen3 REQUIRED)

if (CMAKE_BUILD_TYPE MATCHES Release)
    message("Release Mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64") # tested on Jetson TX2
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+crypto -mcpu=cortex-a57+crypto -flto -ffast-math -fvect-cost-model=unlimited")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch32") # check for Snapdragon Flight Pro
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -flto -ffast-math -fvect-cost-model=unlimited")
    endif ()

    if (${PROFILING} MATCHES false)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-loop-optimizations -fsee -funroll-loops -fno-math-errno -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros")
    endif ()

elseif (CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug Mode")
    add_definitions(-DDEBUG -DDEBUGMSF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    message("Release with Debug Info Mode")
    # Enable asserts
    add_definitions(-UNDEBUG)
endif ()

include_directories(include
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${PHOTOMETRIC_DIRS}
        ${MULTI_UAV_DIRS}
        )

set(SOURCE
        src/x/camera_models/camera_model.cpp
        src/x/camera_models/camera_fov.cpp
        src/x/camera_models/camera_radtan.cpp
        src/x/camera_models/camera_equidistant.cpp
        src/x/camera_models/camera_none.cpp
        src/x/ekf/ekf.cpp
        src/x/ekf/propagator.cpp
        src/x/ekf/state.cpp
        src/x/ekf/state_buffer.cpp
        src/x/ekf/updater.cpp
        src/x/vio/vio.cpp
        src/x/vio/vio_updater.cpp
        src/x/vio/state_manager.cpp
        src/x/vio/track_manager.cpp
        src/x/vio/msckf_update.cpp
        src/x/vio/msckf_slam_update.cpp
        src/x/vio/slam_update.cpp
        src/x/vio/range_update.cpp
        src/x/vio/solar_update.cpp
        src/x/vision/feature.cpp
        src/x/vision/tiled_image.cpp
        src/x/vision/tracker.cpp
        src/x/vision/track.cpp
        src/x/vision/triangulation.cpp
        )

if (MULTI_UAV)
    file(GLOB PR_FILES "src/x/place_recognition/*.cpp")
    set(SOURCE ${SOURCE}
            ${PR_FILES}
            src/x/ekf/ci.cpp
            src/x/ekf/simple_state.cpp
            src/x/vio/multi_slam_update.cpp)
endif ()

if (PHOTOMETRIC_CALI)
    set(SOURCE ${SOURCE} src/x/photometric_calibration/irPhotoCalib.cpp)
endif ()

add_library(x ${SOURCE})

message("PHOTOMETRIC_DIRS: ${PHOTOMETRIC_DIRS}")
message("MULTI_UAV_DIRS: ${MULTI_UAV_DIRS}")

message("PHOTOMETRIC_LIBS: ${PHOTOMETRIC_LIBS}")
message("MULTI_UAV_LIBS: ${MULTI_UAV_LIBS}")
# Additional libraries to link against
set(REQUIRED_LIBRARIES ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${EIGEN3_LIBRARIES}
        ${PHOTOMETRIC_LIBS}
        ${MULTI_UAV_LIBS})

target_link_libraries(x
        ${OpenCV_LIBRARIES};
        ${Boost_LIBRARIES};
        ${EIGEN3_LIBRARIES};
        ${PHOTOMETRIC_LIBS};
        ${MULTI_UAV_LIBS}
        )

# Uninstall Target
# configure_file("${CMAKE_CURRENT_SOURCE_DIR}/xUninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/xUninstall.cmake" IMMEDIATE @ONLY)
# add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/xUninstall.cmake")
set(CONFIG_INCLUDE_DIRS ${CONFIG_INCLUDE_DIRS} "${CMAKE_INSTALL_PREFIX}/include")

# Generate pkg-config file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.pc.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

set(CMAKE_EXPORT_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

include(CMakePackageConfigHelpers)

install(DIRECTORY include/
        DESTINATION include
        )

configure_package_config_file(xConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/xConfig.cmake
        INSTALL_DESTINATION ${CMAKE_EXPORT_DESTINATION}
        )

install(TARGETS x
        EXPORT ${PROJECT_NAME}Exports
        COMPONENT ${PROJECT_NAME}
        DESTINATION ${CMAKE_EXPORT_DESTINATION}
        )

install(EXPORT ${PROJECT_NAME}Exports DESTINATION ${CMAKE_EXPORT_DESTINATION})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/xConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        )

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_EXPORT_DESTINATION}
        COMPONENT ${PROJECT_NAME}
        )

# Support automatic RPM/DEB generation via CPack
set(CPACK_CMAKE_GENERATOR ${CMAKE_GENERATOR})
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_RELEASE ${VERSION})
set(CPACK_PACKAGE_CONTACT "polivicio@gmail.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Vincenzo Polizzi polivicio@gmail.com")
set(CPACK_PACKAGE_VENDOR "JPL-California Institute of Technology")
set(CPACK_PACKAGE_DESCRIPTION "C++ library for multi/single agent vision-based navigation, with multi-sensor fusion capabilities for thermal, range, solar and GPS measurements.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${CPACK_PACKAGE_DESCRIPTION})
set(CPACK_PACKAGING_INSTALL_PREFIX /opt/x/)
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}.${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
set(CPACK_DEB_COMPONENT_INSTALL YES)
set(CPACK_GENERATOR "DEB")
set(CPACK_RPM_PACKAGE_AUTOREQ 1)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS 1)

include(CPack)

message(STATUS "Components to pack: ${CPACK_COMPONENTS_ALL}")
