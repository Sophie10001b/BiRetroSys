#include "main_window.h"

mainWindow::mainWindow(QWidget *parent): QMainWindow(parent){
    this->setWindowTitle("BiRetroSys");
    this->setMinimumSize(QSize(1280, 768));

    auto inputLabel = new QLabel();
    inputLabel->setText("Enter target molecule's SMILES to start route planning");
    inputLabel->setStyleSheet(
        "QLabel{color: black; font: 24px; font-family: Calibri;}"
    );

    this->inputSmi = new QLineEdit();
    this->inputSmi->setPlaceholderText("Target SMILES:");
    this->inputSmi->setStyleSheet(
        "QLineEdit{color: black; border: 1px solid #A0A0A0; border-radius: 20px; padding-left: 20px; selection-background-color: #A0A0A0; font: 20px; font-family: Calibri;}"
        "QLineEdit:hover{border: 1px solid black;}"
    );
    this->inputSmi->setMinimumSize(QSize(700, 50));
    connect(this->inputSmi, &QLineEdit::textEdited, this, &mainWindow::inputSMILES);

    this->searchBtn = new QPushButton();
    setButton(this->searchBtn, "MagnifyingGlass.svg", "  START SEARCH  ");
    this->searchBtn->setStyleSheet(
        "QPushButton{color: white; border-radius: 20px; background-color: #4169E1; font: 20px; font-family: Calibri; text-align: center; background-position: left;}"
        "QPushButton:hover{background-color: #6495ED;}"
        "QPushButton:pressed{background-color: #B0C4DE;}"
    );
    this->searchBtn->setMinimumSize(QSize(220, 50));
    connect(this->searchBtn, &QPushButton::clicked, this, &mainWindow::pushSearchButton);

    this->inputSmiImg = new QSvgWidget();
    this->inputSmiImg->setMinimumSize(500, 125);

    this->setsWidget = new settingsWidget();
    this->mainSideBar = new sideBar();
    QPalette sideBarPalette(this->mainSideBar->palette());
    sideBarPalette.setColor(QPalette::Window, QColor(Qt::white));
    this->mainSideBar->setAutoFillBackground(true);
    this->mainSideBar->setPalette(sideBarPalette);
    this->mainSideBar->setFixedWidth(50);
    
    connect(this->mainSideBar->sideButtonGroup, &QButtonGroup::idClicked, [this](int id){this->mainStackWidget->setCurrentIndex(id);});

    auto searchUpperWidget = new QWidget();
    searchUpperWidget->setStyleSheet("QWidget{border-radius: 5px; background-color: white;}");
    auto searchUpperLayout = new QVBoxLayout();
    auto searchInputLayout = new QHBoxLayout();
    searchInputLayout->addWidget(this->inputSmi);
    searchInputLayout->addWidget(this->searchBtn);
    searchInputLayout->setContentsMargins(0, 0, 0, 0);
    auto mainLogo = new QLabel("BiRetroSys");
    mainLogo->setStyleSheet(
        "QLabel{color: white; font: bold 36px; font-family: Calibri;}"
    );
    searchUpperLayout->addWidget(inputLabel, 1, Qt::AlignCenter);
    searchUpperLayout->addLayout(searchInputLayout, 1);
    searchUpperLayout->addWidget(this->inputSmiImg, 2, Qt::AlignCenter);
    searchUpperWidget->setLayout(searchUpperLayout);
    
    this->searchResultWidget = new QWidget();
    this->searchResultWidget->setStyleSheet("QWidget{border-radius: 5px; background-color: white;}");
    auto searchResultLayout = new QVBoxLayout();
    this->searchResultWidget->setLayout(searchResultLayout);
    this->searchResultWidget->setVisible(false);

    this->searchCurrentText = new QLabel();
    this->searchCurrentText->setStyleSheet("QLabel{color: black; font: 20px; font-family: Calibri;}");
    this->searchProgressText = new QLabel();
    this->searchProgressText->setStyleSheet("QLabel{color: black; font: 20px; font-family: Calibri;}");
    auto searchTextLayout = new QHBoxLayout();
    searchTextLayout->addWidget(this->searchCurrentText, 7, Qt::AlignLeft);
    searchTextLayout->addWidget(this->searchProgressText, 3, Qt::AlignRight);

    this->searchProgress = new QProgressBar();
    this->searchProgress->setStyleSheet(
        "QProgressBar{color: black; height: 16px; text-align: center; font: 12px; font-family: Calibri; border: 0px solid #dcdcdc; border-radius: 5px; background-color: #dcdcdc;}"
        "QProgressBar::chunk{background-color: #cbcbcb; border-radius: 5px;}"
    );
    auto progressLineSplit1 = new QFrame();
    progressLineSplit1->setFrameShape(QFrame::HLine);
    progressLineSplit1->setStyleSheet("QFrame{border: none; border-bottom: 0.5px solid #CCCCCC;}");
    searchResultLayout->addLayout(searchTextLayout, 1);
    searchResultLayout->addWidget(this->searchProgress, 1);
    searchResultLayout->addWidget(progressLineSplit1, 1);

    auto visualizationLayout = new QHBoxLayout();
    auto visualLineSplit1 = new QFrame();
    visualLineSplit1->setFrameShape(QFrame::VLine);
    visualLineSplit1->setStyleSheet("QFrame{border: none; border-left: 0.5px solid #CCCCCC;}");
    auto visualLineSplit2 = new QFrame();
    visualLineSplit2->setFrameShape(QFrame::HLine);
    visualLineSplit2->setStyleSheet("QFrame{border: none; border-bottom: 0.5px solid #CCCCCC;}");
    this->currentMolImg = new QSvgWidget();
    this->currentMolImg->setFixedSize(500, 150);
    // this->currentMolImg->setStyleSheet("QSvgWidget{border: 0.5px solid #cccccc; border-radius: 10px;}");
    auto scoreTopkLayout = new QVBoxLayout();
    auto scoreLayout = new QHBoxLayout();
    auto topkVisualizationLayout = new QHBoxLayout();
    scoreTopkLayout->addLayout(scoreLayout, 2);
    scoreTopkLayout->addWidget(visualLineSplit2, 1);
    scoreTopkLayout->addLayout(topkVisualizationLayout, 8);
    visualizationLayout->addWidget(this->currentMolImg, 6);
    visualizationLayout->addWidget(visualLineSplit1, 1);
    visualizationLayout->addLayout(scoreTopkLayout, 7);
    this->top1Score = new QLabel();
    this->top1Score->setStyleSheet("QLabel{color: black; font: 16px; font-family: Calibri;}");
    this->top1MolImg = new QSvgWidget();
    this->top1MolImg->setFixedSize(150, 150);
    // this->top1MolImg->setStyleSheet("QSvgWidget{background-color: white; border-top-left-radius: 10px; border-top-right-radius: 0px; border-bottom-left-radius: 10px; border-bottom-right-radius: 0px;}");
    this->top2Score = new QLabel();
    this->top2Score->setStyleSheet("QLabel{color: black; font: 16px; font-family: Calibri;}");
    this->top2MolImg = new QSvgWidget();
    this->top2MolImg->setFixedSize(150, 150);
    // this->top2MolImg->setStyleSheet("QSvgWidget{background-color: white;}");
    this->top3Score = new QLabel();
    this->top3Score->setStyleSheet("QLabel{color: black; font: 16px; font-family: Calibri;}");
    this->top3MolImg = new QSvgWidget();
    this->top3MolImg->setFixedSize(150, 150);
    // this->top3MolImg->setStyleSheet("QSvgWidget{background-color: white; border-top-left-radius: 0px; border-top-right-radius: 10px; border-bottom-left-radius: 0px; border-bottom-right-radius: 10px;}");
    scoreLayout->addWidget(this->top1Score, 1, Qt::AlignCenter);
    scoreLayout->addWidget(this->top2Score, 1, Qt::AlignCenter);
    scoreLayout->addWidget(this->top3Score, 1, Qt::AlignCenter);
    topkVisualizationLayout->addWidget(this->top1MolImg, 1);
    topkVisualizationLayout->addWidget(this->top2MolImg, 1);
    topkVisualizationLayout->addWidget(this->top3MolImg, 1);
    searchResultLayout->addLayout(visualizationLayout, 4);

    auto searchLayout = new QVBoxLayout();
    searchLayout->addWidget(mainLogo, 2, Qt::AlignCenter);
    searchLayout->addWidget(searchUpperWidget, 6);
    searchLayout->addWidget(this->searchResultWidget, 6);
    auto searchWidget = new QWidget();
    searchWidget->setLayout(searchLayout);
    this->mainStackWidget = new QStackedWidget();
    this->mainStackWidget->addWidget(searchWidget);
    this->mainStackWidget->addWidget(this->setsWidget);
    this->mainStackWidget->setCurrentIndex(0);
    this->mainStackWidget->setStyleSheet("QWidget{background-color: gray;}");

    this->mainLayout = new QHBoxLayout();
    this->mainLayout->addWidget(this->mainSideBar);
    this->mainLayout->addWidget(this->mainStackWidget);
    this->mainLayout->setContentsMargins(0, 0, 0, 0);
    this->mainLayout->setSpacing(0);

    QWidget *centralWidget = new QWidget();
    centralWidget->setLayout(this->mainLayout);
    setCentralWidget(centralWidget);
}

mainWindow::~mainWindow(){}

str mainWindow::getMolSVGImg(const str &smi, int width, int height){
    auto mol = RDKit::SmilesToMol(smi, 0, false);
    str molSVG;
    if (mol){
        RDKit::MolDraw2DSVG molDrawer(width, height);
        molDrawer.setLineWidth(1.0);
        // RDKit::assignBWPalette(molDrawer.drawOptions().atomColourPalette);
        molDrawer.drawMolecule(*mol);
        molDrawer.finishDrawing();
        molSVG = molDrawer.getDrawingText();
    }
    else{molSVG = "";}
    return molSVG;
}

void mainWindow::loadTerminalMols(){
    this->loadDiag = new loadingDialog();
    this->loadThread = new QThread();
    this->loadDiag->loadWorker->moveToThread(this->loadThread);

    connect(this->loadThread, &QThread::started, this->loadDiag->loadWorker, [this](){this->loadDiag->loadWorker->loadTerminalMols(this->molSet);});
    connect(this->loadThread, &QThread::finished, loadThread, &QThread::deleteLater);
    connect(this->loadDiag->loadWorker, &loadingWorker::loadingFinished, loadThread, &QThread::quit);
    connect(this->loadDiag->loadWorker, &loadingWorker::loadingFinished, this, [this](){this->loadDiag->close(); this->show();});
    connect(this->loadDiag->loadWorker, &loadingWorker::loadingCount, this->loadDiag, &loadingDialog::updateProgress);

    connect(this->loadDiag->loadWorker, &loadingWorker::loadingFailure, this, [this](str loadPath){
        str msg = "Terminal molecules loading failure, from path:\n" + loadPath;
        auto msgBox = QMessageBox::critical(this, "Terminal Molecules", msg.c_str());
        this->loadDiag->close();
        this->close();
    });

    this->loadDiag->show();
    this->loadThread->start();
}

void mainWindow::pushSearchButton(){
    if (!this->isSearchButtonClicked){
        if (RDKit::SmilesToMol(this->inputSmi->text().toStdString(), 0, false)){
            this->searchResultWidget->setVisible(true);
            setButton(this->searchBtn, "NoSymbol.svg", "NOW SEARCHING...");
            this->searchBtn->setStyleSheet(
                "QPushButton{color: white; border-radius: 20px; background-color: #1E90FF; font: 20px; font-family: Calibri; text-align: center; background-position: left;}"
                "QPushButton:hover{background-color: #6495ED;}"
                "QPushButton:pressed{background-color: #B0C4DE;}"
            );
            this->inputSmi->setReadOnly(true);
            this->multiStepSearch();
            this->isSearchButtonClicked = !this->isSearchButtonClicked;
        }
        else{
            auto msgBox = QMessageBox::critical(this, "Invalid Inputs", "Current input SMILES can not be recognized by RDKit, please check your input SMILES !");
        }
    }
    else{
        setButton(this->searchBtn, "MagnifyingGlass.svg", "  START SEARCH  ");
        this->searchBtn->setStyleSheet(
            "QPushButton{color: white; border-radius: 20px; background-color: #4169E1; font: 20px; font-family: Calibri; text-align: center; background-position: left;}"
            "QPushButton:hover{background-color: #6495ED;}"
            "QPushButton:pressed{background-color: #B0C4DE;}"
        );
        this->inputSmi->setReadOnly(false);
        if (this->searchT && this->searchT->runSynthesis){
            this->searchT->runSynthesis = false;
        }
        this->isSearchButtonClicked = !this->isSearchButtonClicked;
    }
}

void mainWindow::inputSMILES(const QString &smi){
    str targetSmi = smi.toStdString();
    auto targetSmiSVG = this->getMolSVGImg(targetSmi, 400, 100);
    if (!targetSmiSVG.empty()){
        this->inputSmiImg->load(QByteArray::fromStdString(targetSmiSVG));
        this->inputSmiImg->resize(500, 125);
    }
}

void mainWindow::multiStepSearch(){
    this->searchThread = new QThread();
    this->searchT = new multiStepSynthesis(
        this->inputSmi->text().toStdString(),
        &this->molSet,
        this->setsWidget->getSettings()
    );
    this->searchT->moveToThread(this->searchThread);

    connect(this->searchThread, &QThread::started, this->searchT, &multiStepSynthesis::startSearch);
    connect(this->searchThread, &QThread::started, this, [this](){
        this->searchProgress->setStyleSheet(
            "QProgressBar{color: black; height: 16px; text-align: center; font: 12px; font-family: Calibri; border: 0px solid #dcdcdc; border-radius: 5px; background-color: #dcdcdc;}"
            "QProgressBar::chunk{background-color: #cbcbcb; border-radius: 5px;}"
        );
    });
    connect(this->searchThread, &QThread::finished, this->searchThread, &QThread::deleteLater);
    connect(this->searchT, &multiStepSynthesis::finishSearch, this, [this](int status, multiStepSynthesisLog searchStatus){
        switch (status){
            case 0: {
                this->searchCurrentText->setText(QString("Search Finishing! TOTAL STEPS: %1, TIMES: %2").arg(searchStatus.spendSteps).arg(searchStatus.spendTimes, 0, 'g', 4)); this->searchT->tree->finishSearch();
                str msg;
                if (this->searchT->tree->hasFound){
                    msg = "Search Success! Route and Tree have been saved at:\n" + this->searchT->tree->savePath;
                    auto msgBox = QMessageBox::information(this, "Search Finished", msg.c_str());
                }
                else{
                    msg = "Search Failure! Tree have been saved at:\n" + this->searchT->tree->savePath;
                    auto msgBox = QMessageBox::warning(this, "Search Finished", msg.c_str());
                }
                break;
            }

            case 1: this->searchCurrentText->setText(QString("No Open Nodes, Stop Searching! TOTAL STEPS: %1, TIMES: %2").arg(searchStatus.spendSteps).arg(searchStatus.spendTimes, 0, 'g', 4)); this->searchT->tree->finishSearch(); break;

            case 2: this->searchCurrentText->setText(QString("Search is stopped manually")); break;
        }
        delete this->searchT;
        this->searchT = nullptr;
        this->searchThread->quit();

        setButton(this->searchBtn, "MagnifyingGlass.svg", "  START SEARCH  ");
        this->searchBtn->setStyleSheet(
            "QPushButton{color: white; border-radius: 20px; background-color: #4169E1; font: 20px; font-family: Calibri; text-align: center; background-position: left;}"
            "QPushButton:hover{background-color: #6495ED;}"
            "QPushButton:pressed{background-color: #B0C4DE;}"
        );
        this->inputSmi->setReadOnly(false);
        this->isSearchButtonClicked = false;
    });
    connect(this->searchT, &multiStepSynthesis::findRoute, this, [this](){
        this->searchProgress->setStyleSheet(
            "QProgressBar{color: black; height: 16px; text-align: center; font: 12px; font-family: Calibri; border: 0px solid #dcdcdc; border-radius: 5px; background-color: #dcdcdc;}"
            "QProgressBar::chunk{background-color: #439cf4; border-radius: 5px;}"
        );
    });
    connect(this->searchT, &multiStepSynthesis::finishEachStep, this, [this](int curStep, int totalStep, const str &curSmi, std::vector<std::vector<str>> topSmi, std::vector<float> topSmiScore){
        this->searchProgressText->setText(QString("%1 / %2").arg(curStep).arg(totalStep));
        this->searchCurrentText->setText(QString("Trying to Expand %1").arg(curSmi.c_str()));
        this->searchProgress->setValue(curStep);

        // sort
        for (int i=0, sz=topSmiScore.size(); i < std::min(sz, 3); i++){
            for (int j=sz-1; j > i; j--){
                if (topSmiScore[j] > topSmiScore[j-1]){
                    std::swap(topSmiScore[j], topSmiScore[j-1]);
                    std::swap(topSmi[j], topSmi[j-1]);
                }
            }
        }

        std::vector<str> top3Reaction;
        std::vector<float> top3ReactionScore;
        for (int i=0; i < 3; i++){
            if (i >= topSmi.size()){
                top3Reaction.push_back("");
                top3ReactionScore.push_back(0);
            }
            else{
                if (topSmi[i].size() > 1){
                    str smis = "";
                    for (auto &s : topSmi[i]) smis += s + '.';
                    if (smis.back() == '.') smis.pop_back();
                    top3Reaction.push_back(smis);
                    
                }
                else{
                    top3Reaction.push_back(topSmi[i][0]);
                }
                top3ReactionScore.push_back(topSmiScore[i]);
            }
        }

        auto curMol = this->getMolSVGImg(curSmi, 400, 120);
        if (!curMol.empty()){
            this->currentMolImg->load(QByteArray::fromStdString(curMol));
            this->currentMolImg->resize(500, 150);
        }
        auto top1Mol = this->getMolSVGImg(top3Reaction[0], 150, 150);
        if (!top1Mol.empty()){
            this->top1MolImg->load(QByteArray::fromStdString(top1Mol));
            this->top1MolImg->resize(150, 150);
            this->top1Score->setText(QString("%1%").arg(top3ReactionScore[0] * 100, 0, 'g', 4));
        }
        auto top2Mol = this->getMolSVGImg(top3Reaction[1], 150, 150);
        if (!top2Mol.empty()){
            this->top2MolImg->load(QByteArray::fromStdString(top2Mol));
            this->top2MolImg->resize(150, 150);
            this->top2Score->setText(QString("%1%").arg(top3ReactionScore[1] * 100, 0, 'g', 4));
        }
        auto top3Mol = this->getMolSVGImg(top3Reaction[2], 150, 150);
        if (!top3Mol.empty()){
            this->top3MolImg->load(QByteArray::fromStdString(top3Mol));
            this->top3MolImg->resize(150, 150);
            this->top3Score->setText(QString("%1%").arg(top3ReactionScore[2] * 100, 0, 'g', 4));
        }
    });

    this->searchThread->start();
}