#pragma once

#include "include_head.h"

struct searchSettings{
    bool isEarlyStop;
    bool needConsistCheck;
    int expansionWidth;
    int checkWidth;
    int singleSearchSteps;
    int multiSearchSteps;
    float temperature;
    float expansionLowerBound;
    float checkLowerBound;

    searchSettings(bool earlyStop, bool consistCheck, int expansionW, int checkW, int singleS, int multiS, float T, float expansionL, float checkL): isEarlyStop(earlyStop), needConsistCheck(consistCheck), expansionWidth(expansionW), checkWidth(checkW), singleSearchSteps(singleS), multiSearchSteps(multiS), temperature(T), expansionLowerBound(expansionL), checkLowerBound(checkL){}
};

class settingsWidget: public QWidget{
    Q_OBJECT

    public:
    settingsWidget(QWidget *parent=nullptr);
    ~settingsWidget();

    searchSettings getSettings();

    signals:
    void settingsChanged();

    private:
    QVBoxLayout *settingsLayout = nullptr;
    QLabel *curTitle = nullptr;
    QCheckBox *earlyStop = nullptr;
    QCheckBox *consistCheck = nullptr;
    QSlider *expansionWidth = nullptr;
    QSlider *checkWidth = nullptr;
    QSlider *singleSteps = nullptr;
    QSlider *multiSteps = nullptr;
    QSlider *temperature = nullptr;
    QSlider *expansionLowerBound = nullptr;
    QSlider *checkLowerBound = nullptr;

    const QString sliderCSS = {
        "QSlider::groove:horizontal{height: 30px;}"
        "QSlider::sub-page:horizontal{background: #439cf4; border-radius: 3px; margin-top: 8px; margin-bottom: 8px;}"
        "QSlider::add-page:horizontal{background: #dcdcdc; border-radius: 3px; margin-top: 10px; margin-bottom: 10px;}"
        "QSlider::handle:horizontal{background: #cbcbcb; border: none; border-radius: 5px; margin-top: -10px; margin-bottom: -10px;}"
        "QSlider::handle:horizontal:hover{background: #cbcbcb; border: 1px solid black; border-radius: 5px; margin-top: -10px; margin-bottom: -10px;}"
    };
};