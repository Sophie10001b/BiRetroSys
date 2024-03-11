#pragma once
#include <MolHandler/include_head.h>

namespace MolHandler {
    //atom features
    constexpr int ATOMFEATNUM = 13;

    const initlist<str> ATOMLIST = {"C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "W", "Ru", "Nb", "Re", "Te", "Rh", "Ta", "Tc", "Ba", "Bi", "Hf", "Mo", "U", "Sm", "Os", "Ir", "Ce", "Gd", "Ga", "Cs", "<unk>"};

    constexpr initlist<RDKit::Atom::HybridizationType> HYBRIDLIST = {RDKit::Atom::HybridizationType::SP, RDKit::Atom::HybridizationType::SP2, RDKit::Atom::HybridizationType::SP3, RDKit::Atom::HybridizationType::SP3D, RDKit::Atom::HybridizationType::SP3D2};

    constexpr initlist<RDKit::Atom::ChiralType> CHIRALIST = {RDKit::Atom::ChiralType::CHI_TETRAHEDRAL_CW, RDKit::Atom::ChiralType::CHI_TETRAHEDRAL_CCW, RDKit::Atom::ChiralType::CHI_UNSPECIFIED};

    const initlist<str> RSLIST = {"R", "S", "NONE"};

    constexpr initlist<int> NDEGREE = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    constexpr initlist<int> NCHARGE = {-1, -2, 1, 2, 0};
    constexpr initlist<int> NVALENCE = {0, 1, 2, 3, 4, 5, 6};
    constexpr initlist<int> NHS = {0, 1, 3, 4, 5};


    template <typename T1, typename T2>
    inline std::map<T2, int> ilist2map(const T1 &ilist){
        std::map<T2, int> temp;
        int i = 0;
        for (auto elem : ilist){
            temp.insert(std::make_pair(elem, i));
            ++i;
        }
        return temp;
    }

    template <typename T1, typename T2>
    inline T2 mapGet(const std::map<T1, T2> &tgtMap, const T1 &tgt, const T2 &udef){
        auto res = tgtMap.find(tgt);
        T2 matchRes = res == tgtMap.end() ? udef : res->second;
        return matchRes;
    }

    const auto atomMap = ilist2map<const initlist<str>, str>(ATOMLIST);

    const auto hybridMap = ilist2map<const initlist<RDKit::Atom::HybridizationType>, RDKit::Atom::HybridizationType>(HYBRIDLIST);

    const auto chiralMap = ilist2map<const initlist<RDKit::Atom::ChiralType>, RDKit::Atom::ChiralType>(CHIRALIST);

    const auto rsMap = ilist2map<const initlist<str>, str>(RSLIST);

    const auto degreeMap = ilist2map<const initlist<int>, int>(NDEGREE);

    const auto chargeMap = ilist2map<const initlist<int>, int>(NCHARGE);

    const auto valenceMap = ilist2map<const initlist<int>, int>(NVALENCE);

    const auto hsMap = ilist2map<const initlist<int>, int>(NHS);

    const std::vector<int> ATOMFEATDIM = {int(ATOMLIST.size()), int(NDEGREE.size()), int(NCHARGE.size()), int(NVALENCE.size()), int(NHS.size()), int(CHIRALIST.size()), int(RSLIST.size()), int(HYBRIDLIST.size()), 2, 10, 2, 2, 10};

    template <typename T>
    inline void getAtomFeat(
        const RDKit::Atom &atom, T *container, const int lTask=0, const int lClass=-1, const int lRoot=0, const int lShuffle=0
    ){
        auto atomSymbol = atomMap.find(atom.getSymbol());
        if (atomSymbol == atomMap.end()){*(container) = atomMap.at("<unk>");}
        else {
            *(container) = atomSymbol->second;
            *(container + 1) = mapGet(degreeMap, int(atom.getDegree()), 9);
            *(container + 2) = mapGet(chargeMap, atom.getFormalCharge(), 4);
            *(container + 3) = mapGet(valenceMap, int(atom.getTotalValence()), 6);
            *(container + 4) = mapGet(hsMap, int(atom.getTotalNumHs()), 4);
            *(container + 5) = mapGet(chiralMap, atom.getChiralTag(), 2);
            
            str rsTag = "";
            if (atom.getPropIfPresent("_CIPCode", rsTag)){*(container + 6) = mapGet(rsMap, rsTag, 2);}
            else {*(container + 6) = 2;}

            *(container + 7) = mapGet(hybridMap, atom.getHybridization(), 4);
            *(container + 8) = atom.getIsAromatic();

            auto tempIt = {lRoot, lShuffle, lTask, lClass};
            std::copy(tempIt.begin(), tempIt.end(), container + 9);
        }
    }

    //bond features
    constexpr int BONDFEATNUM = 4;

    constexpr initlist<RDKit::Bond::BondType> BONDLIST = {RDKit::Bond::BondType::SINGLE, RDKit::Bond::BondType::DOUBLE, RDKit::Bond::BondType::TRIPLE, RDKit::Bond::BondType::AROMATIC};

    constexpr initlist<RDKit::Bond::BondStereo> STEREOLIST = {RDKit::Bond::BondStereo::STEREONONE, RDKit::Bond::BondStereo::STEREOE, RDKit::Bond::BondStereo::STEREOZ};

    const auto bondMap = ilist2map<const initlist<RDKit::Bond::BondType>, RDKit::Bond::BondType>(BONDLIST);

    const auto stereoMap = ilist2map<const initlist<RDKit::Bond::BondStereo>, RDKit::Bond::BondStereo>(STEREOLIST);

    const std::vector<int> BONDFEATDIM = {int(BONDLIST.size()), int(STEREOLIST.size()), 2, 2};

    template <typename T>
    inline void getBondFeat(const RDKit::Bond &bond, const RDKit::RingInfo &ringInfo, T *container){
        auto bondType = bondMap.find(bond.getBondType());
        if (bondType != bondMap.end()){
            *(container) = bondType->second;
            *(container + 1) = mapGet(stereoMap, bond.getStereo(), 0);
            *(container + 2) = bond.getIsConjugated();
            *(container + 3) = bool(ringInfo.numBondRings(bond.getIdx()));
        }
    }
}