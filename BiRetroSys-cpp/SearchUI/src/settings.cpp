#include "settings.h"

settingsWidget::settingsWidget(QWidget *parent): QWidget(parent){
    QFont titleFont("Calibri", 28);
    QFont labelFont("Calibri", 16);
    titleFont.setBold(true);
    this->settingsLayout = new QVBoxLayout();
    this->setLayout(this->settingsLayout);
    auto normalSettingsLayout = new QGridLayout();
    normalSettingsLayout->setContentsMargins(0, 5, 0, 5);
    auto expansionSettingsLayout = new QGridLayout();
    expansionSettingsLayout->setContentsMargins(0, 5, 0, 5);
    auto checkSettingsLayout = new QGridLayout();
    checkSettingsLayout->setContentsMargins(0, 5, 0, 5);

    auto normalSettingsWidget = new QWidget();
    normalSettingsWidget->setStyleSheet("QWidget{border-radius: 5px; background-color: white;}");
    normalSettingsWidget->setLayout(normalSettingsLayout);
    auto expansionSettingsWidget = new QWidget();
    expansionSettingsWidget->setStyleSheet("QWidget{border-radius: 5px; background-color: white;}");
    expansionSettingsWidget->setLayout(expansionSettingsLayout);
    auto checkSettingsWidget = new QWidget();
    checkSettingsWidget->setStyleSheet("QWidget{border-radius: 5px; background-color: white;}");
    checkSettingsWidget->setLayout(checkSettingsLayout);

    this->curTitle = new QLabel("Settings");
    this->curTitle->setStyleSheet("QLabel{color: white; font: 32px bold; font-family: Calibri;}");
    this->settingsLayout->addWidget(this->curTitle, 1);

    // Search settings
    auto normalSettingsTitle = new QLabel("Search");
    normalSettingsTitle->setStyleSheet("QLabel{color: black; font: 28px; font-family: Calibri;}");
    auto normalSplitLine1 = new QFrame();
    normalSplitLine1->setFrameShape(QFrame::HLine);
    normalSplitLine1->setStyleSheet("QFrame{border: none; border-bottom: 0.5px solid #CCCCCC;}");
    auto normalSplitLine2 = new QFrame();
    normalSplitLine2->setFrameShape(QFrame::HLine);
    normalSplitLine2->setStyleSheet("QFrame{border: none; border-bottom: 0.5px solid #CCCCCC;}");
    this->earlyStop = new QCheckBox("EarlyStop");
    this->earlyStop->setStyleSheet("QCheckBox{color: black; font: 18px; font-family: Calibri;}");
    this->earlyStop->setChecked(true);
    this->consistCheck = new QCheckBox("ConsistCheck");
    this->consistCheck->setStyleSheet("QCheckBox{color: black; font: 18px; font-family: Calibri;}");
    this->consistCheck->setChecked(true);
    
    auto singleStepsLabel = new QLabel("Single Step Search Steps:");
    singleStepsLabel->setStyleSheet("QLabel{color: black; font: 18px; font-family: Calibri;}");
    this->singleSteps = new QSlider(Qt::Horizontal);
    this->singleSteps->setRange(1, 20); // (50, 1000)
    this->singleSteps->setSingleStep(1);
    this->singleSteps->setValue(3);
    this->singleSteps->setStyleSheet(this->sliderCSS);
    connect(this->singleSteps, &QSlider::sliderMoved, [](int value){
        QToolTip::showText(QCursor::pos(), QString("%1").arg(value * 50));
    });
    auto multiStepsLabel = new QLabel("Multi Step Search Steps:");
    multiStepsLabel->setStyleSheet("QLabel{color: black; font: 18px; font-family: Calibri;}");
    this->multiSteps = new QSlider(Qt::Horizontal);
    this->multiSteps->setRange(1, 10); // (50, 500)
    this->multiSteps->setSingleStep(1);
    this->multiSteps->setValue(2);
    this->multiSteps->setStyleSheet(this->sliderCSS);
    connect(this->multiSteps, &QSlider::sliderMoved, [](int value){
        QToolTip::showText(QCursor::pos(), QString("%1").arg(value * 50));
    });

    auto temperatureLabel = new QLabel("Temperature(T):");
    temperatureLabel->setStyleSheet("QLabel{color: black; font: 18px; font-family: Calibri;}");
    this->temperature = new QSlider(Qt::Horizontal);
    this->temperature->setRange(1, 20); // (0.1, 2.0)
    this->temperature->setSingleStep(1);
    this->temperature->setValue(10);
    this->temperature->setStyleSheet(this->sliderCSS);
    connect(this->temperature, &QSlider::sliderMoved, [](int value){
        QToolTip::showText(QCursor::pos(), QString("%1").arg(float(value)/10.0, 0, 'g', 2));
    });

    normalSettingsLayout->addWidget(normalSettingsTitle, 0, 1, 2, 9, Qt::AlignLeft);
    normalSettingsLayout->addWidget(normalSplitLine1, 1, 0, 1, 10);
    normalSettingsLayout->addWidget(this->earlyStop, 2, 1, 1, 4);
    normalSettingsLayout->addWidget(this->consistCheck, 2, 5, 1, 4);
    normalSettingsLayout->addWidget(normalSplitLine2, 3, 0, 1, 10);
    normalSettingsLayout->addWidget(singleStepsLabel, 4, 1, 1, 4);
    normalSettingsLayout->addWidget(multiStepsLabel, 4, 5, 1, 4);
    normalSettingsLayout->addWidget(this->singleSteps, 5, 1, 1, 4);
    normalSettingsLayout->addWidget(this->multiSteps, 5, 5, 1, 4);
    normalSettingsLayout->addWidget(temperatureLabel, 6, 1, 1, 4);
    normalSettingsLayout->addWidget(this->temperature, 7, 1, 1, 4);
    this->settingsLayout->addWidget(normalSettingsWidget, 5);

    // Expansion settings
    auto expansionTitle = new QLabel("Retrosynthesis");
    expansionTitle->setStyleSheet("QLabel{color: black; font: 28px; font-family: Calibri;}");
    auto expansionSplitLine1 = new QFrame();
    expansionSplitLine1->setFrameShape(QFrame::HLine);
    expansionSplitLine1->setStyleSheet("QFrame{border: none; border-bottom: 0.5px solid #CCCCCC;}");
    auto expansionWidthLabel = new QLabel("ExpansionWidth:");
    expansionWidthLabel->setStyleSheet("QLabel{color: black; font: 18px; font-family: Calibri;}");
    this->expansionWidth = new QSlider(Qt::Horizontal);
    this->expansionWidth->setRange(10, 30);
    this->expansionWidth->setSingleStep(5);
    this->expansionWidth->setValue(20);
    this->expansionWidth->setStyleSheet(this->sliderCSS);
    connect(this->expansionWidth, &QSlider::sliderMoved, [&](int value){
        QToolTip::showText(QCursor::pos(), QString("%1").arg(value));
    });
    auto expansionLowerBoundLabel = new QLabel("Expansion Filter:");
    expansionLowerBoundLabel->setStyleSheet("QLabel{color: black; font: 18px; font-family: Calibri;}");
    this->expansionLowerBound = new QSlider(Qt::Horizontal);
    this->expansionLowerBound->setRange(0, 20); // (0.0, 0.2)
    this->expansionLowerBound->setSingleStep(1);
    this->expansionLowerBound->setValue(10);
    this->expansionLowerBound->setStyleSheet(this->sliderCSS);
    connect(this->expansionLowerBound, &QSlider::sliderMoved, [](int value){
        QToolTip::showText(QCursor::pos(), QString("%1").arg(float(value)/100.0, 0, 'g', 2));
    });

    expansionSettingsLayout->addWidget(expansionTitle, 0, 1, 2, 9, Qt::AlignLeft);
    expansionSettingsLayout->addWidget(expansionSplitLine1, 1, 0, 1, 10);
    expansionSettingsLayout->addWidget(expansionWidthLabel, 2, 1, 1, 4);
    expansionSettingsLayout->addWidget(expansionLowerBoundLabel, 2, 5, 1, 4);
    expansionSettingsLayout->addWidget(expansionWidth, 3, 1, 1, 4);
    expansionSettingsLayout->addWidget(expansionLowerBound, 3, 5, 1, 4);
    this->settingsLayout->addWidget(expansionSettingsWidget, 3);

    // Check settings
    auto checkTitle = new QLabel("Forward Synthesis");
    checkTitle->setStyleSheet("QLabel{color: black; font: 28px; font-family: Calibri;}");
    auto checkSplitLine1 = new QFrame();
    checkSplitLine1->setFrameShape(QFrame::HLine);
    checkSplitLine1->setStyleSheet("QFrame{border: none; border-bottom: 0.5px solid #CCCCCC;}");
    auto checkWidthLabel = new QLabel("CheckWidth:");
    checkWidthLabel->setStyleSheet("QLabel{color: black; font: 18px; font-family: Calibri;}");
    this->checkWidth = new QSlider(Qt::Horizontal);
    this->checkWidth->setRange(10, 30);
    this->checkWidth->setSingleStep(5);
    this->checkWidth->setValue(20);
    this->checkWidth->setStyleSheet(this->sliderCSS);
    connect(this->checkWidth, &QSlider::sliderMoved, [](int value){
        QToolTip::showText(QCursor::pos(), QString("%1").arg(value));
    });
    auto checkLowerBoundLabel = new QLabel("ConsistCheck Filter:");
    checkLowerBoundLabel->setStyleSheet("QLabel{color: black; font: 18px; font-family: Calibri;}");
    this->checkLowerBound = new QSlider(Qt::Horizontal);
    this->checkLowerBound->setRange(0, 20); // (0.0, 0.2)
    this->checkLowerBound->setSingleStep(1);
    this->checkLowerBound->setValue(1);
    this->checkLowerBound->setStyleSheet(this->sliderCSS);
    connect(this->checkLowerBound, &QSlider::sliderMoved, [](int value){
        QToolTip::showText(QCursor::pos(), QString("%1").arg(float(value)/100.0, 0, 'g', 2));
    });

    checkSettingsLayout->addWidget(checkTitle, 0, 1, 2, 9, Qt::AlignLeft);
    checkSettingsLayout->addWidget(checkSplitLine1, 1, 0, 1, 10);
    checkSettingsLayout->addWidget(checkWidthLabel, 2, 1, 1, 4);
    checkSettingsLayout->addWidget(checkLowerBoundLabel, 2, 5, 1, 4);
    checkSettingsLayout->addWidget(checkWidth, 3, 1, 1, 4);
    checkSettingsLayout->addWidget(checkLowerBound, 3, 5, 1, 4);
    this->settingsLayout->addWidget(checkSettingsWidget, 3);
}

settingsWidget::~settingsWidget(){}

searchSettings settingsWidget::getSettings(){
    return searchSettings(
        this->earlyStop->isChecked(),
        this->consistCheck->isChecked(),
        this->expansionWidth->value(),
        this->checkWidth->value(),
        this->singleSteps->value() * 50,
        this->multiSteps->value() * 50,
        float(this->temperature->value()) / 10.0,
        float(this->expansionLowerBound->value()) / 100.0,
        float(this->checkLowerBound->value()) / 100.0
    );
}