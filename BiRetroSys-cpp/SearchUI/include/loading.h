#pragma once

#include "include_head.h"

class loadingWorker: public QObject{
    Q_OBJECT

    public:
    loadingWorker(QObject *parent=nullptr);
    ~loadingWorker();

    const str molNums = "23081629";
    str loadPath;

    public slots:
    void loadTerminalMols(std::unordered_set<str> &molSet);

    signals:
    void loadingFinished();
    void loadingFailure(str loadPath);
    void loadingCount(int num);
};

class loadingDialog: public QDialog{
    Q_OBJECT

    public slots:
    void updateProgress(int num);

    public:
    loadingDialog(QWidget *parent=nullptr);
    ~loadingDialog();

    loadingWorker *loadWorker = nullptr;

    private:
    QLabel *loadLog = nullptr;
    QLabel *loadProgress = nullptr;
    QProgressBar *loadBar = nullptr;
    QGridLayout *loadLayout = nullptr;
    QTimer *updateTimer = nullptr;
};

