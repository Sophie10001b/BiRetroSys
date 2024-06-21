#pragma once

#include "include_head.h"
#include "loading.h"
#include "settings.h"
#include "side_bar.h"
#include "synthesis.h"

class mainWindow : public QMainWindow{
    Q_OBJECT
    
    public:
    mainWindow(QWidget *parent=nullptr);
    ~mainWindow();
    void loadTerminalMols();

    private slots:
    void inputSMILES(const QString &smi);
    void pushSearchButton();

    private:
    std::unordered_set<str> molSet;

    loadingDialog *loadDiag = nullptr;
    sideBar *mainSideBar = nullptr;
    settingsWidget *setsWidget = nullptr;
    bool isSideBarClicked = false;
    bool isSearchButtonClicked = false;

    QLineEdit *inputSmi = nullptr;
    QPushButton *searchBtn = nullptr;
    QWidget *searchResultWidget = nullptr;
    QSvgWidget *inputSmiImg = nullptr;
    QSvgWidget *currentMolImg = nullptr;
    QProgressBar *searchProgress = nullptr;
    QLabel *searchCurrentText = nullptr;
    QLabel *searchProgressText = nullptr;
    QLabel *top1Score = nullptr;
    QSvgWidget *top1MolImg = nullptr;
    QLabel *top2Score = nullptr;
    QSvgWidget *top2MolImg = nullptr;
    QLabel *top3Score = nullptr;
    QSvgWidget *top3MolImg = nullptr;

    QHBoxLayout *mainLayout = nullptr;
    QStackedWidget *mainStackWidget = nullptr;

    QThread *loadThread = nullptr;

    multiStepSynthesis *searchT = nullptr;
    QThread *searchThread = nullptr;

    void multiStepSearch();
    str getMolSVGImg(const str &smi, int width, int height);
};