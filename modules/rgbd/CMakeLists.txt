set(the_description "RGBD algorithms")

if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()

ocv_define_module(rgbd opencv_core opencv_calib3d opencv_imgproc OPTIONAL opencv_viz WRAP python)

if(NOT HAVE_EIGEN)
  message(STATUS "rgbd: Eigen support is disabled. Eigen is Required for Posegraph optimization")
endif()

if(HAVE_OPENGL)
  ocv_target_link_libraries(${the_module} PRIVATE "${OPENGL_LIBRARIES}")
endif()
