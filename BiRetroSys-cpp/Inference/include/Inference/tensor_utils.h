#pragma once
#include <Inference/include_head.h>

namespace Inference {
    template <typename T>
    inline int64_t numel(const std::vector<T> &shape){
        int64_t Size = 1;
        for (const auto &i : shape){Size *= i;}
        return Size;
    }

    // template <typename T>
    // inline int64_t numel(const T *shape, const int64_t shapeCount){
    //     int64_t Size = 1;
    //     for (int64_t i=0; i < shapeCount; i++){Size *= *(shape + i);}
    //     return Size;
    // }

    template <typename T>
    inline void tensorOutput(const T *data, const std::vector<int64_t> &shape){
        for (int i=0; i < numel(shape); i++){
            std::cout << *(data + i) << "\t";
        }
        std::cout << "------------" << std::endl;
    }
    
    template <typename Src, typename Tgt>
    inline Ort::Value convertTensor(MatRX<Src> &srcMat, Ort::MemoryInfo &memInfo, const std::vector<int64_t> &assumSize={}){
        std::vector<int64_t> matSize;
        if (assumSize.size() != 0) matSize = assumSize;
        else{
            if (srcMat.rows() > 1){matSize.push_back(srcMat.rows());}
            if (srcMat.cols() > 1 || srcMat.cols() == 1 && srcMat.rows() == 1){matSize.push_back(srcMat.cols());}
        }
        return Ort::Value::CreateTensor<Tgt>(
            memInfo, srcMat.data(), srcMat.size(),
            matSize.data(), matSize.size()
        );
    }

    template <typename Src, typename Tgt>
    inline Ort::Value convertTensor(MatRX<Src> &srcMat, std::vector<int64_t> &matSize, Ort::MemoryInfo &memInfo){
        return Ort::Value::CreateTensor<Tgt>(
            memInfo, srcMat.data(), srcMat.size(),
            matSize.data(), matSize.size()
        );
    }

    template <typename Src, typename Tgt>
    inline Ort::Value convertTensor(Src *srcMat, std::vector<int64_t> &matShape, Ort::MemoryInfo &memInfo){
        auto size = numel(matShape);
        return Ort::Value::CreateTensor<Tgt>(
            memInfo, srcMat, size,
            matShape.data(), matShape.size()
        );
    }

    // template <typename Src, typename Tgt>
    // inline std::vector<Ort::Value> convertTensor(std::vector<MatRX<Src>> &srcMats, Ort::MemoryInfo &memInfo){
    //     std::vector<Ort::Value> res;
    //     std::vector<int64_t> matSize;
    //     for (auto srcMat : srcMats){
    //         matSize = {};
    //         int count = 1;
    //         if (srcMat.rows() > 1){matSize.push_back(srcMat.rows()); count *= srcMat.rows();}
    //         if (srcMat.cols() > 1){matSize.push_back(srcMat.cols()); count *= srcMat.cols();}
    //         Ort::Value tensor = Ort::Value::CreateTensor<Tgt>(
    //                 memInfo, srcMat.data(), srcMat.size(),
    //                 matSize.data(), matSize.size()
    //             );
    //         // const int64_t* tensorPtr = tensor.GetTensorData<int64_t>();
    //         // for (int i=0; i < count; i++){
    //         //     std::cout << *(tensorPtr + i) << "\t";
    //         // }
    //         // std::cout << "\n" << "--------------" << std::endl;
    //         res.push_back(
    //             Ort::Value::CreateTensor<Tgt>(
    //                 memInfo, srcMat.data(), srcMat.size(),
    //                 matSize.data(), matSize.size()
    //             )
    //         );
    //     }
    //     return res;
    // }

    template <typename T1, typename T2>
    inline MatRX<T2> convert(MatRX<T1> &mat){
        MatRX<T2> res = MatRX<T2>(mat.rows(), mat.cols());
        auto resPtr = res.data();
        auto matPrt = mat.data();
        for (int i=0; i < mat.size(); i++){*(resPtr + i) = static_cast<T2>(*(matPrt + i));}
        return res;
    }

    template <typename T>
    inline std::vector<MatRX<T>> graphPadding(const T *mat, const std::vector<int64_t> &shape, const MatRX<int64_t> &graphLength){
        auto maxLength = graphLength.maxCoeff();
        int64_t batchSize = graphLength.size();
        int64_t dModel = shape[1];
        std::vector<MatRX<T>> padMats(batchSize);
        MatRX<T> padMat;
        int64_t nodeNum;
        int64_t preNodeNum = 0;

        for (int i=0; i < batchSize; i++){
            nodeNum = graphLength(0, i);
            padMat = MatRX<T>::Zero(maxLength, dModel);
            std::copy(mat + preNodeNum * dModel, mat + preNodeNum * dModel + nodeNum * dModel, padMat.data());
            padMats[i] = padMat;
            preNodeNum += nodeNum;
        }
        return padMats;
    }

    inline std::vector<MatRX<bool>> getMask(
        const int queryLength, const int keyLength,
        const MatRX<int64_t> &validLength
    ){
        int64_t batchSize = validLength.size();
        std::vector<MatRX<bool>> masks(batchSize);
        MatRX<bool> mask;
        // MatRX<bool> onesMat = MatRX<bool>::Ones(queryLength, keyLength);
        for (int i=0; i < batchSize; i++){
            mask = MatRX<bool>::Zero(queryLength, keyLength);
            // mask.leftCols(validLength(0, i)) = onesMat.leftCols(validLength(0, i));
            mask.leftCols(validLength(0, i)).fill(true);
            masks[i] = mask;
        }
        return masks;
    }

    template <typename T>
    inline T *batchRepeatInterleave(const T *data, std::vector<int64_t> &shape, const int64_t repeatNum, bool del=true){
        int64_t count = 1;
        int64_t batchSize = shape[0];
        for (int i=1; i < shape.size(); i++){count *= shape[i];}
        // T *repeatData = (T *)malloc(repeatNum * sizeof(T) * count * batchSize);
        T *repeatData = new T[repeatNum * batchSize * count];

        #pragma omp parallel for
        for (int i=0; i < batchSize; i++){
            for (int j=0; j < repeatNum; j++){
                std::copy(data + i * count, data + (i + 1) * count, repeatData + j * count + i * repeatNum * count);
            }
        }
        shape[0] *= repeatNum;
        if (del){delete []data;}
        return repeatData;
    }

    template <typename T1, typename T2>
    inline T2 *batchRepeatInterleave(const std::vector<T1> &data, std::vector<int64_t> &shape, const int64_t repeatNum){
        int64_t count = 1;
        int64_t batchSize = shape[0];
        for (int i=1; i < shape.size(); i++){count *= shape[i];}
        // T2 *repeatData = (T2 *)malloc(repeatNum * sizeof(T2) * count * batchSize);
        T2 *repeatData = new T2[repeatNum * batchSize * count];

        #pragma omp parallel for
        for (int i=0; i < batchSize; i++){
            for (int j=0; j < repeatNum; j++){
                std::copy(data[i].data(), data[i].data() + count, repeatData + j * count + i * repeatNum * count);
            }
        }
        shape[0] *= repeatNum;
        return repeatData;
    }

    template <typename T>
    inline std::vector<int64_t> indexGenerate(const T &boolIndex, bool condition=true){
        std::vector<int64_t> index;
        for (int i=0; i < boolIndex.size(); i++){
            if ((boolIndex[i] && condition) || (!boolIndex[i] && !condition)){index.push_back(i);}
        }
        return index;
    }

    template <typename T>
    inline std::tuple<T*, std::vector<int64_t>> indexSelect(const T *data, const std::vector<int64_t> &shape, const std::vector<int64_t> &index, int64_t dim=0, bool del=true){
        dim = dim < 0 ? shape.size() + dim : dim;
        int64_t repeatCount = 1;
        int64_t copySizeCount = 1;
        int64_t repeatSizeCount = 1;
        for (int i=0; i < shape.size(); i++){
            if (i < dim){repeatCount *= shape[i];}
            if (i > dim){copySizeCount *= shape[i];}
            if (i >= dim){repeatSizeCount *= shape[i];}
        }
        for (auto i : index){assert(i < shape[dim]);}

        // T * indexData = (T*)malloc(repeatCount * index.size() * copySizeCount * sizeof(T));
        auto idxCount = index.size();
        T *indexData = new T[repeatCount * idxCount * copySizeCount];

        #pragma omp parallel for
        for (int i=0; i < repeatCount; i++){
            for (int j=0; j < idxCount; j++){
                std::copy(data + i * repeatSizeCount + index[j] * copySizeCount, data + i * repeatSizeCount + (index[j] + 1) * copySizeCount, indexData + i * copySizeCount * idxCount + j * copySizeCount);
            }
        }
        std::vector<int64_t> newShape = shape;
        newShape[dim] = idxCount;
        if (del){delete []data;}
        return std::make_tuple(indexData, newShape);
    }

    template <typename T>
    inline T *indexSelect(const T *data, std::vector<int64_t> &shape, const std::vector<int64_t> &index, int64_t dim=0, bool del=true){
        dim = dim < 0 ? shape.size() + dim : dim;
        int64_t repeatCount = 1;
        int64_t copySizeCount = 1;
        int64_t repeatSizeCount = 1;
        for (int i=0; i < shape.size(); i++){
            if (i < dim){repeatCount *= shape[i];}
            if (i > dim){copySizeCount *= shape[i];}
            if (i >= dim){repeatSizeCount *= shape[i];}
        }
        for (auto i : index){assert(i < shape[dim]);}

        // T * indexData = (T*)malloc(repeatCount * index.size() * copySizeCount * sizeof(T));
        auto idxCount = index.size();
        T *indexData = new T[repeatCount * idxCount * copySizeCount];

        #pragma omp parallel for
        for (int i=0; i < repeatCount; i++){
            for (int j=0; j < idxCount; j++){
                std::copy(data + i * repeatSizeCount + index[j] * copySizeCount, data + i * repeatSizeCount + (index[j] + 1) * copySizeCount, indexData + i * copySizeCount * idxCount + j * copySizeCount);
            }
        }
        shape[dim] = idxCount;
        if (del){delete []data;}
        return indexData;
    }

    template <typename T>
    inline std::tuple<std::vector<T>, std::vector<int64_t>> indexSelect(const std::vector<T> &data, const std::vector<int64_t> &shape, const std::vector<int64_t> &index, int64_t dim=0){
        dim = dim < 0 ? shape.size() + dim : dim;
        int64_t repeatCount = 1;
        int64_t copySizeCount = 1;
        int64_t repeatSizeCount = 1;
        for (int i=0; i < shape.size(); i++){
            if (i < dim){repeatCount *= shape[i];}
            if (i > dim){copySizeCount *= shape[i];}
            if (i >= dim){repeatSizeCount *= shape[i];}
        }
        for (auto i : index){assert(i < shape[dim]);}

        // T * indexData = (T*)malloc(repeatCount * index.size() * copySizeCount * sizeof(T));
        auto idxCount = index.size();
        std::vector<T> indexData(repeatCount * idxCount * copySizeCount);

        #pragma omp parallel for
        for (int i=0; i < repeatCount; i++){
            for (int j=0; j < idxCount; j++){
                std::copy(data.begin() + i * repeatSizeCount + index[j] * copySizeCount, data.begin() + i * repeatSizeCount + (index[j] + 1) * copySizeCount, indexData.begin() + i * copySizeCount * idxCount + j * copySizeCount);
            }
        }
        std::vector<int64_t> newShape = shape;
        newShape[dim] = idxCount;
        return std::make_tuple(indexData, newShape);
    }

    template <typename T>
    inline std::vector<T> indexSelect(const std::vector<T> &data, std::vector<int64_t> &shape, const std::vector<int64_t> &index, int64_t dim=0){
        dim = dim < 0 ? shape.size() + dim : dim;
        int64_t repeatCount = 1;
        int64_t copySizeCount = 1;
        int64_t repeatSizeCount = 1;
        for (int i=0; i < shape.size(); i++){
            if (i < dim){repeatCount *= shape[i];}
            if (i > dim){copySizeCount *= shape[i];}
            if (i >= dim){repeatSizeCount *= shape[i];}
        }
        for (auto i : index){assert(i < shape[dim]);}

        // T * indexData = (T*)malloc(repeatCount * index.size() * copySizeCount * sizeof(T));
        auto idxCount = index.size();
        std::vector<T> indexData(repeatCount * idxCount * copySizeCount);

        #pragma omp parallel for
        for (int i=0; i < repeatCount; i++){
            for (int j=0; j < idxCount; j++){
                std::copy(data.begin() + i * repeatSizeCount + index[j] * copySizeCount, data.begin() + i * repeatSizeCount + (index[j] + 1) * copySizeCount, indexData.begin() + i * copySizeCount * idxCount + j * copySizeCount);
            }
        }
        shape[dim] = idxCount;
        return indexData;
    }

    template <typename T>
    inline std::vector<T> firstIndexSelect(const std::vector<T> &data, const std::vector<int64_t> &index){
        auto idxCount = index.size();
        std::vector<T> indexData(idxCount);

        #pragma omp parallel for
        for (int i=0; i < idxCount; i++){indexData[i] = data[index[i]];}
        return indexData;
    }

    template <typename T>
    inline void indexCopy(T *data, const T *copyData, const std::vector<int64_t> &shape, const std::vector<int64_t> &index){
        int64_t dim = 0;
        int64_t copySizeCount = 1;
        for (int i=0; i < shape.size(); i++){
            if (i > dim){copySizeCount *= shape[i];}
        }
        for (auto i : index){assert(i < shape[dim]);}
        auto idxCount = index.size();

        #pragma omp parallel for
        for (int i=0; i < idxCount; i++){
            std::copy(copyData + i * copySizeCount, copyData + (i + 1) * copySizeCount, data + index[i] * copySizeCount);
        }
    }

    template <typename T>
    inline void indexCopy(std::vector<std::vector<T>> &data, const std::vector<std::vector<T>> &copyData, const std::vector<int64_t> &index){
        int64_t dim = 0;
        int64_t copySizeCount = data[0].size();
        for (const auto &i : index){assert(i < data.size());}
        auto idxCount = index.size();

        #pragma omp parallel for
        for (int i=0; i < idxCount; i++){
            data[index[i]] = copyData[i];
        }
    }

    // template <typename T>
    // inline T *indexSelect(const T *data, std::vector<int64_t> &shape, const std::vector<bool> &index, int64_t dim=0, bool del=true, bool condition=true){
    //     dim = dim < 0 ? shape.size() + dim : dim;
    //     int64_t repeatCount = 1;
    //     int64_t copySizeCount = 1;
    //     int64_t repeatSizeCount = 1;
    //     for (int i=0; i < shape.size(); i++){
    //         if (i < dim){repeatCount *= shape[i];}
    //         if (i > dim){copySizeCount *= shape[i];}
    //         if (i >= dim){repeatSizeCount *= shape[i];}
    //     }
    //     assert(index.size() == shape[dim]);
    //     int64_t selectCount = condition ? std::accumulate(index.begin(), index.end(), 0) : index.size() - std::accumulate(index.begin(), index.end(), 0);

    //     // T * indexData = (T*)malloc(repeatCount * index.size() * copySizeCount * sizeof(T));
    //     T *indexData = new T[repeatCount * selectCount * copySizeCount];
    //     for (int i=0; i < repeatCount; i++){
    //         for (int j=0, idxCount=0; j < index.size(); j++){
    //             if ((index[j] && condition) || (!index[j] && !condition)){
    //                 std::copy(data + i * repeatSizeCount + j * copySizeCount, data + i * repeatSizeCount + (j + 1) * copySizeCount, indexData + i * copySizeCount * selectCount + idxCount * copySizeCount);
    //                 idxCount++;
    //             }
    //         }
    //     }
    //     shape[dim] = selectCount;
    //     if (del){delete []data;}
    //     return indexData;
    // }

    // template <typename T>
    // inline void indexCopy(T *data, const T *copyData, std::vector<int64_t> &shape, const std::vector<bool> &index, bool condition=true){
    //     int64_t dim = 0;
    //     int64_t copySizeCount = 1;
    //     for (int i=0; i < shape.size(); i++){
    //         if (i > dim){copySizeCount *= shape[i];}
    //     }
    //     assert(index.size() == shape[dim]);
    //     int64_t selectCount = condition ? std::accumulate(index.begin(), index.end(), 0) : index.size() - std::accumulate(index.begin(), index.end(), 0);

    //     for (int i=0, idxCount=0; i < index.size(); i++){
    //         if ((index[i] && condition) || (!index[i] && !condition)){
    //             std::copy(copyData + idxCount * copySizeCount, copyData + (idxCount + 1) * copySizeCount, data + i * copySizeCount);
    //             idxCount++;
    //         }
    //     }
    // }

    template <typename T>
    inline T * tensorCopy(const T *data, const int64_t size, bool del=true){
        T *copyData = new T[size];
        std::copy(data, data + size, copyData);
        if (del){delete []data;}
        return copyData;
    }

    inline std::vector<int64_t> binCount(const int64_t *data, const int64_t size, const int64_t minLength=1){
        std::vector<int64_t> counts;
        std::unordered_map<int64_t, int64_t> countMap;
        for (int i=0; i < size; i++){
            auto mapping = countMap.find(*(data + i));
            if (mapping != countMap.end()){
                mapping->second += 1;
            }
            else {countMap.insert(std::make_pair(*(data + i), 1));}
        }
        for (auto i : countMap){counts.push_back(i.second);}
        for (int i=0; i < minLength - counts.size(); i++){counts.push_back(0);}
        return counts;
    }

    template <typename T>
    inline std::vector<int64_t> constBinCount(const T *data, const int64_t size, const std::vector<T> &labels){
        std::vector<int64_t> counts(labels.size());
        std::unordered_map<T, int64_t> countMap;
        for (int i=0; i < labels.size(); i++){countMap.insert(std::make_pair(labels[i], 0LL));}

        for (int i=0; i < size; i++){
            auto mapping = countMap.find(*(data + i));
            if (mapping != countMap.end()){
                mapping->second += 1;
            }
        }
        for (int i=0; i < labels.size(); i++){
            counts[i] = countMap.find(labels[i])->second;
        }
        return counts;
    }

    //--------------------------------------------------
    template <typename T>
    inline void lastSoftmax(T *data, const std::vector<int64_t> &shape, const float temperature=1.0, bool needLog=false){
        int64_t dimSize = shape.back();
        int64_t sizes = 1;
        for (int i=0; i < shape.size() - 1; i ++){sizes *= shape[i];}
        int64_t count = sizes * dimSize;

        #if (defined _OPENMP) && (_OPPAL)
        #pragma omp parallel for
        for (int i=0; i < count; i++){data[i] = std::exp(data[i] / temperature);}

        std::vector<T> sumData(sizes, 0);
        
        for (int i=0; i < sizes; i++){
            T tempSum = 0;
            #pragma omp parallel for reduction(+ : tempSum)
            for (int j=0; j < dimSize; j++){tempSum += *(data + i * dimSize + j);}
            sumData[i] = tempSum;
        }

        if (needLog){
            #pragma omp parallel for
            for (int i=0; i < sizes; i++){
                for (int j=0; j < dimSize; j++){
                    *(data + i * dimSize + j) = std::log(*(data + i * dimSize + j) / sumData[i]);
                }
            }
        }
        else {
            #pragma omp parallel for
            for (int i=0; i < sizes; i++){
                for (int j=0; j < dimSize; j++){
                    *(data + i * dimSize + j) /= sumData[i];
                }
            }
        }
        #else
        T sumData = 0;
        std::for_each(data, data + count, [&temperature](T &a){a = std::exp(a / temperature);});
        for (int i=0; i < sizes; i++){
            sumData = std::reduce(data + i * dimSize, data + (i + 1) * dimSize, T(0));
            if (needLog){
                std::for_each(data + i * dimSize, data + (i + 1) * dimSize, [&sumData](T &a){a = std::log(a / sumData);});
            }
            else {std::for_each(data + i * dimSize, data + (i + 1) * dimSize, [&sumData](T &a){a /= sumData;});}
        }
        #endif
    }

    template <typename T>
    inline std::tuple<std::vector<T>, std::vector<int64_t>> lastTopK(T *data, const std::vector<int64_t> &shape, const int64_t topk, bool largest=true, bool del=true){
        int64_t copyCount = shape.back();
        assert(topk <= copyCount);
        int64_t repeatCount = 1;
        for (int i=0; i < shape.size() - 1; i++){repeatCount *= shape[i];}

        std::vector<T> topkData(repeatCount * topk);
        std::vector<int64_t> topkIdx(repeatCount * topk);

        auto __less = [](const T &a, const T &b){return a < b;};
        auto __greater = [](const T &a, const T &b){return a > b;};

        // heap sort
        auto __createHeap = [&__less, &__greater](std::vector<T> &tgt, std::vector<int64_t> &tgtIdx, int64_t begin, int64_t end, const bool largest=true) -> void{
            int64_t parent = begin;
            int64_t child = parent * 2 + 1;
            auto __cmp = largest ? __greater : __less;
            while (child <= end){
                if (child + 1 <= end && __cmp(tgt[child+1], tgt[child])) child++;
                if (!__cmp(tgt[child], tgt[parent])) break;
                std::swap(tgt[child], tgt[parent]);
                std::swap(tgtIdx[child], tgtIdx[parent]);
                parent = child; child = child * 2 + 1;
            }
        };

        auto __cmp = largest ? __greater : __less;
        #pragma omp parallel for
        for (int i=0; i < repeatCount; i++){
            std::vector<T> tempData(topk);
            std::vector<int64_t> tempIdx(topk);
            std::copy(data + i * copyCount, data + i * copyCount + topk, tempData.begin());
            std::iota(tempIdx.begin(), tempIdx.end(), 0);

            for (int k=topk/2-1; k >= 0; k--) __createHeap(tempData, tempIdx, k, topk-1, !largest);
            for (int k=topk; k < copyCount; k++){
                if (__cmp(data[i*copyCount+k], tempData[0])){
                    tempData[0] = data[i*copyCount+k];
                    tempIdx[0] = k;
                    __createHeap(tempData, tempIdx, 0, topk-1, !largest);
                }
            }

            // final sort
            for (int k=topk/2-1; k >= 0; k--) __createHeap(tempData, tempIdx, k, topk-1, !largest);
            for (int k=topk-1; k >= 0; k--){
                std::swap(tempData[0], tempData[k]);
                std::swap(tempIdx[0], tempIdx[k]);
                __createHeap(tempData, tempIdx, 0, k-1, !largest);
            }
            std::copy(tempData.begin(), tempData.end(), topkData.begin() + i * topk);
            std::copy(tempIdx.begin(), tempIdx.end(), topkIdx.begin() + i * topk);
        }

        // #pragma omp parallel for
        // for (int i=0; i < repeatCount; i++){
        //     std::vector<T> tempData(copyCount);
        //     std::vector<int64_t> tempIdx(copyCount);
        //     std::copy(data + i * copyCount, data + (i + 1) * copyCount, tempData.begin());
        //     std::iota(tempIdx.begin(), tempIdx.end(), 0);
        //     for (int k=0; k < topk; k++){
        //         for (int j=tempData.size() - 1; j > k; j--){
        //             if ((tempData[j] > tempData[j-1] && largest) || (tempData[j] < tempData[j-1] && !largest)){
        //                 std::swap(tempData[j], tempData[j-1]);
        //                 std::swap(tempIdx[j], tempIdx[j-1]);
        //             }
        //         }
        //     }
        //     std::copy(tempData.begin(), tempData.begin() + topk, topkData.begin() + i * topk);
        //     std::copy(tempIdx.begin(), tempIdx.begin() + topk, topkIdx.begin() + i * topk);
        // }
        if (del){delete []data;}
        return std::make_tuple(topkData, topkIdx);
    }
}