if (AMReX_SPACEDIM GREATER_EQUAL 3)
    add_executable(shock_cloud cloud.cpp ${QuokkaObjSources})
    add_executable(test_grackle_cooling test_grackle_cooling.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(shock_cloud)
        setup_target_for_cuda_compilation(test_grackle_cooling)
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
    
endif(AMReX_SPACEDIM GREATER_EQUAL 3)
