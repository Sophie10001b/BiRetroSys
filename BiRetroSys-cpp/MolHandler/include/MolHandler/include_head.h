#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <initializer_list>
#include <regex>
#include <chrono>
#include <numeric>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/FileParsers/MolWriters.h>

#include <Eigen/Core>

using str = std::string;

template<typename T>
using initlist = std::initializer_list<T>;

template<typename T>
using MatRX = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;

template<typename T>
using VecRX = Eigen::Matrix<T, 1, -1, Eigen::RowMajor>;