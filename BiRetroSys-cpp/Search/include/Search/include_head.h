#pragma once
#include <queue>
#include <graphviz/gvc.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>

#include <Inference/model_utils.h>
#include <Inference/tensor_utils.h>

#ifdef _OPENMP
    #define _RDPAL false
#endif

inline str getLogTime(){
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto localT = localtime(&now);
    auto logTime = std::put_time(localT, "[%Y-%m-%d : %H:%M:%S]");   
    std::ostringstream oss;
    oss << logTime;
    return oss.str();
}

inline void outputLog(const str &contain, std::ofstream &tgtLog, const bool toTerminal=true){
    str outputText = getLogTime() + "  " + contain + "\n";
    tgtLog << outputText;
    std::cout << outputText;
}