function (enable_intellisense TARGET_NAME)
    # Hack to get intellisense working for CUDA includes    
    set_target_cuda_arch_flags(${TARGET_NAME})
    add_test(NAME  ${TARGET_NAME} COMMAND ${TARGET_NAME})

    cmake_path(SET path2 "${DIR}")
    cmake_path(GET path2 FILENAME DIR_NAME)
    set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER benchmarks/${DIR_NAME})
    add_cuda_to_target(${TARGET_NAME} "")
    
    if(${ENABLE_DEBUG})
        add_cuda_debug_support_to_target(${TARGET_NAME})
    endif()

    if(${ENABLE_NVTX})
        add_nvtx_support_to_target(${TARGET_NAME})
    endif()

    if(${ENABLE_BENCHMARK})
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_BENCHMARK)
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
  
    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_SOURCE_DIR}")
    add_optimization_flags(${TARGET_NAME})
endfunction()

function (discover_tests DIR)
    file(
        GLOB_RECURSE
        CUDA_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.cpp"
        "${DIR}/*.cu"
    )
    
    foreach(cuda_source ${CUDA_SOURCES})
		get_filename_component(cuda_target ${cuda_source} NAME_WE)       
		add_executable(${cuda_target} ${cuda_source} ${LAUNCH_SOURCES})
		add_fkl_to_target(${cuda_target})
        target_link_libraries(${cuda_target} PRIVATE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig) 								              
		target_include_directories(${cuda_target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
		enable_intellisense(${cuda_target})
    endforeach()
endfunction()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

MACRO(SUBDIRLIST result curdir)
FILE(GLOB children ${curdir}/*) # 
SET(dirlist "")
FOREACH(child ${children})
    IF(IS_DIRECTORY ${child}) 
        LIST(APPEND dirlist ${child})
    ENDIF()
ENDFOREACH()
SET(${result} ${dirlist} )
ENDMACRO()

set (LIST_OF_DIRS "")
SUBDIRLIST(LIST_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
 
foreach(DIR ${LIST_DIRS})
    discover_tests(${DIR})    
endforeach()