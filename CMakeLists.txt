cmake_minimum_required(VERSION 3.22)

project(form
	VERSION 0.1.0
	LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS Off)

option(FORM_BUILD_TESTS "Build FORM Tests" OFF)
option(FORM_BUILD_PYTHON "Build python evalio wrapper" OFF)

# ------------------------- Dependencies ------------------------- #
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(TBB REQUIRED)

include(FetchContent)
FetchContent_Declare(
    tsl
    GIT_REPOSITORY https://github.com/Tessil/robin-map.git
    GIT_TAG v1.4.0
    GIT_SHALLOW TRUE
)
# Populate but don't add as subdirectory (header-only library)
FetchContent_GetProperties(tsl)
if(NOT tsl_POPULATED)
    FetchContent_Populate(tsl)
endif()

# ------------------------- form ------------------------- #
add_library(FORM
	form/form.cpp
	form/feature/factor.cpp
	form/optimization/constraints.cpp
	form/mapping/keyscanner.cpp
)
target_include_directories(FORM
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:include>
	PRIVATE
	${tsl_SOURCE_DIR}/include
)
# Explicitly list header files for IDE support
set(FORM_HEADERS
	# feature
	form/feature/factor.hpp
	form/feature/features.hpp
	form/feature/extraction.hpp
	# optimization
	form/optimization/constraints.hpp
	form/optimization/gtsam.hpp
	# mapping
	form/mapping/map.hpp
	form/mapping/keyscanner.hpp
	# core
	form/form.hpp
	form/utils.hpp
)
target_sources(FORM PRIVATE ${FORM_HEADERS})
target_link_libraries(FORM
	PUBLIC
	gtsam
	Eigen3::Eigen
	TBB::tbb
)

# Tests
if(${FORM_BUILD_TESTS})
	add_subdirectory(tests)
endif()

# python bindings
if(${FORM_BUILD_PYTHON})
	add_subdirectory(python)
endif()

# Install library
install(TARGETS FORM
	EXPORT FORMTargets
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY form/
	DESTINATION include/form
	FILES_MATCHING PATTERN "*.hpp" PATTERN "*.tpp"
)

# Create and install CMake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/formConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/formConfig.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/formConfig.cmake"
	INSTALL_DESTINATION lib/cmake/form
)

install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/formConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/formConfigVersion.cmake"
	DESTINATION lib/cmake/form
)

install(EXPORT FORMTargets
	FILE formTargets.cmake
	NAMESPACE form::
	DESTINATION lib/cmake/form
)