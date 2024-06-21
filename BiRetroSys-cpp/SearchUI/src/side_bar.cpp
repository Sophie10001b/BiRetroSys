#include "side_bar.h"

QCustomSideButton::QCustomSideButton(QWidget *parent): QPushButton(parent){}

void QCustomSideButton::enterEvent(QEvent *event){
    QToolTip::showText(QCursor::pos(), this->toolTip());
}

QCustomSideToolButton::QCustomSideToolButton(QWidget *parent): QToolButton(parent){}

void QCustomSideToolButton::enterEvent(QEvent *event){
    QToolTip::showText(QCursor::pos(), this->toolTip());
}

sideBar::sideBar(QWidget *parent): QWidget(parent){
    this->sideBarButton = new QCustomSideButton();
    setButton(this->sideBarButton, "Bars3.svg", "");
    this->sideBarButton->setFixedSize(50, 50);
    this->sideBarButton->setStyleSheet(
        "QPushButton{border: none;}"
        "QPushButton:hover{background-color: #C0C0C0;}"
        "QPushButton:pressed{background-color: #CCCCCC;}"
    );
    this->sideBarButton->setToolTip("Menu");

    this->sideButtonGroup = new QButtonGroup();
    this->sideButtonGroup->setExclusive(true);
    this->searchButton = new QCustomSideToolButton();
    setButton(this->searchButton, "Share.svg", "");
    this->searchButton->setFixedSize(50, 50);
    this->searchButton->setStyleSheet(
        "QToolButton{border: none;}"
        "QToolButton:hover{background-color: #C0C0C0;}"
        "QToolButton:pressed, QToolButton:checked{background-color: #CCCCCC;}"
    );
    this->searchButton->setToolTip("Multi-Step Search");
    this->searchButton->setCheckable(true);

    this->settingButton = new QCustomSideToolButton();
    setButton(this->settingButton, "Cog6Tooth.svg", "");
    this->settingButton->setFixedSize(50, 50);
    this->settingButton->setStyleSheet(
        "QToolButton{border: none;}"
        "QToolButton:hover{background-color: #C0C0C0;}"
        "QToolButton:pressed, QToolButton:checked{background-color: #CCCCCC;}"
    );
    this->settingButton->setCheckable(true);

    this->sideButtonGroup->addButton(this->searchButton, 0);
    this->sideButtonGroup->addButton(this->settingButton, 1);
    this->sideButtonGroup->button(0)->setChecked(true);

    this->sideBarLayout = new QVBoxLayout();
    this->sideBarLayout->addWidget(this->sideBarButton, 1);
    this->sideBarLayout->addWidget(this->searchButton, 1);
    this->sideBarLayout->addWidget(this->settingButton, 1);
    this->sideBarLayout->addStretch(7);
    this->sideBarLayout->setContentsMargins(0, 0, 0, 0);
    this->sideBarLayout->setSpacing(0);

    this->setLayout(this->sideBarLayout);
}

sideBar::~sideBar(){}