#include "loading.h"

loadingWorker::loadingWorker(QObject *parent): QObject(parent){
    this->loadPath = std::filesystem::current_path().parent_path();
    this->loadPath += "/Models/origin_dict.csv";
}

loadingWorker::~loadingWorker(){}

void loadingWorker::loadTerminalMols(std::unordered_set<str> &molSet){
    auto loadBegin = std::chrono::high_resolution_clock::now();
    std::ifstream terminalMols;

    if (std::filesystem::exists(this->loadPath)){
        terminalMols.open(this->loadPath, std::ios::in);
        str line;
        std::getline(terminalMols, line);

        int loadNum = 0;
        while (std::getline(terminalMols, line)){
            int start = 0, end=line.size() - 1;
            for (; start < end && line[start] != ','; start++){}
            for (; start < end && line[end] != ','; end--){}
            if (start < end){
                molSet.insert(line.substr(start+1, end-start-1));
                emit this->loadingCount(++loadNum);
            }
            // if (loadNum == 1000) break;
        }
        terminalMols.close();
        auto loadEnd = std::chrono::high_resolution_clock::now();
        auto loadCost = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadBegin).count() * 1e-3;
        emit this->loadingFinished();
    }
    else{
        emit this->loadingFailure(this->loadPath);
    }
    
}

loadingDialog::loadingDialog(QWidget *parent): QDialog(parent){
    enFont(12);
    this->setFont(enfont);
    this->setWindowTitle("BiRetroSys Loading...");

    this->loadWorker = new loadingWorker();

    str logText = "Loading Terminal Molecules from " + this->loadWorker->loadPath;
    this->loadLog = new QLabel();
    this->loadLog->setText(logText.c_str());
    this->loadLog->setStyleSheet("QLabel{font: 16px; font-family: Calibri;}");

    str loadProgress = "Total Molecules: " + this->loadWorker->molNums;
    this->loadProgress = new QLabel();
    this->loadProgress->setText(loadProgress.c_str());
    this->loadProgress->setStyleSheet("QLabel{font: 16px; font-family: Calibri;}");

    this->loadBar = new QProgressBar();
    this->loadBar->setTextVisible(true);
    this->loadBar->setRange(0, std::atoi(this->loadWorker->molNums.c_str()));
    this->loadBar->setStyleSheet(
        "QProgressBar{color: black; height: 16px; text-align: center; font: 12px; font-family: Calibri; border-radius: 5px; background-color: #DCDCDC;}"
        "QProgressBar::chunk{background-color: #A9A9A9; border-radius: 5px;}"
    );

    this->loadLayout = new QGridLayout(this);
    this->loadLayout->addWidget(this->loadLog, 0, 0);
    this->loadLayout->addWidget(this->loadBar, 1, 0);
    this->loadLayout->addWidget(this->loadProgress, 2, 0);
}

loadingDialog::~loadingDialog(){
    delete this->loadWorker;
}

void loadingDialog::updateProgress(int num){
    this->loadBar->setValue(num);
}