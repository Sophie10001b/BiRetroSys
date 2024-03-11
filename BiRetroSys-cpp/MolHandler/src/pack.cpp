#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <MolHandler/chem_utils.h>
#include <MolHandler/data_utils.h>

using Arg = pybind11::arg;

using namespace MolHandler {
    PYBIND11_MODULE(MolHandler, m){
        m.doc() = "A C++ Preprocess Module for Input Molecules";
        m.attr("distBlock") = DISTBLOCK;
        m.attr("atomDim") = std::accumulate(ATOMFEATDIM.begin(), ATOMFEATDIM.end(), 0);
        m.attr("bondDim") = std::accumulate(BONDFEATDIM.begin(), BONDFEATDIM.end(), 0);
        pybind11::class_<inputData>(m, "inputData")
            .def(pybind11::init<const int&, const int&>(), Arg("num"), Arg("k"))
            .def_readwrite("atomFeat", &inputData::atomFeat)
            .def_readwrite("bondFeat", &inputData::bondFeat)
            .def_readwrite("bondIdx", &inputData::bondIdx)
            .def_readwrite("attnBondIdx", &inputData::attnBondIdx)
            .def_readwrite("queryIdx", &inputData::queryIdx)
            .def_readwrite("keyIdx", &inputData::keyIdx)
            .def_readwrite("deg", &inputData::deg)
            .def_readwrite("dist", &inputData::dist)
            .def_readwrite("seqFeat", &inputData::seqFeat)
            .def_readwrite("lTask", &inputData::lTask)
            .def_readwrite("lClass", &inputData::lClass)
            .def_readwrite("graphLength", &inputData::graphLength)
            .def_readwrite("seqLength", &inputData::seqLength)
            .def_readwrite("bondSplit", &inputData::bondSplit);
        
        pybind11::class_<molPreprocess>(m, "molPreprocess")
            .def(pybind11::init<const int, const int, const int>(), Arg("maxDeg"), Arg("maxK"), Arg("maxPath"))
            .def(pybind11::init<const str&, const int, const int, const int>(), Arg("vdir"), Arg("maxDeg"), Arg("maxK"), Arg("maxPath"))
            .def_readonly("maxDeg", &molPreprocess::maxDeg)
            .def_readonly("maxK", &molPreprocess::maxK)
            .def_readonly("maxPath", &molPreprocess::maxPath)
            .def_readwrite("vocab", &molPreprocess::vocab)
            .def("readVocabulary", &molPreprocess::readVocabulary, Arg("vdir"), Arg("extraToken"))
            .def("canonicalizeSmiles", &molPreprocess::canonicalizeSmiles, Arg("smi"))
            .def("generateSeq", &molPreprocess::generateSeq, Arg("smi"), Arg("vocab"))
            .def("generateGraph", &molPreprocess::generateGraph, Arg("smi"), Arg("inData"), Arg("preCumLength"), Arg("preLength"), Arg("batchId"), Arg("ltask"), Arg("lClass"), Arg("lRoot"), Arg("lShuffle"))
            .def("getQKVIndex", &molPreprocess::getQKVIndex, Arg("qIdx"), Arg("kvIdx"), Arg("preLength"), Arg("curLength"))
            .def("getBatchBond", &molPreprocess::getBatchBond, Arg("bondFeat"), Arg("curBondFeat"), Arg("bondIdx"), Arg("curBondIdx"), Arg("attnBondIdx"), Arg("preCumLength"), Arg("preLength"), Arg("curLength"))
            .def("getPaddingSeq", &molPreprocess::getPaddingSeq, Arg("unpadSeq"), Arg("seqFeat"), Arg("seqLength"), Arg("maxSeqLength"), Arg("needBos")=false, Arg("vocab"))
            .def("generateBatch", &molPreprocess::generateBatch, Arg("smis"), Arg("lTasks"), Arg("needSeq"))
            .def("getAttentionBias", &molPreprocess::getAttentionBias, Arg("mol"))
            .def("getKhopFeat", &molPreprocess::getKhopFeat, Arg("mol"));
    }
};