#pragma once

#include "include_head.h"

class QCustomSideButton: public QPushButton{
    Q_OBJECT

    public:
    QCustomSideButton(QWidget *parent=nullptr);
    
    protected:
    void enterEvent(QEvent *event);
};

class QCustomSideToolButton: public QToolButton{
    Q_OBJECT

    public:
    QCustomSideToolButton(QWidget *parent=nullptr);

    protected:
    void enterEvent(QEvent *event);
};

class sideBar: public QWidget{
    Q_OBJECT

    public:
    sideBar(QWidget *parent=nullptr);
    ~sideBar();

    QCustomSideButton *sideBarButton = nullptr;
    QCustomSideToolButton *searchButton = nullptr;
    QCustomSideToolButton *settingButton = nullptr;
    QButtonGroup *sideButtonGroup = nullptr;
    QVBoxLayout *sideBarLayout = nullptr;
};