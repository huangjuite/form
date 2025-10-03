# Setup gtest
find_package(GTest REQUIRED)
enable_testing() # enable ctest

add_executable(form-tests
    test_PlanarFactor.cpp
    test_PointFactor.cpp
    test_MixedFactor.cpp
    test_SeparateFactor.cpp
    test_ClassicFactor.cpp
)
target_link_libraries(form-tests 
    GTest::gtest_main
    FORM 
)
include(GoogleTest)
gtest_discover_tests(form-tests)

# Make runnable from "make check" or "make test"
add_custom_target(check COMMAND form-tests)
