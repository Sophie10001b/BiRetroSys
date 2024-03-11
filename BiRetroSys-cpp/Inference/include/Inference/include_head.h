#pragma once
#include <filesystem>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>

#include <MolHandler/data_utils.h>

#ifdef _OPENMP
    #include <omp.h>
    #define _OPPAL false
#endif