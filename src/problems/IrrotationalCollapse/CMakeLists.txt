if(AMReX_SPACEDIM EQUAL 3)
  set(JOB_NAME IrrotationalCollapse)

  add_executable(
    ${JOB_NAME} testIrrotationalCollapse.cpp ../../turbulence/TurbDataReader.cpp
                ${QuokkaObjSources})
  if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
  endif()
endif()
