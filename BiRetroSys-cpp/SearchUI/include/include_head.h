#pragma once

#include <QThread>
#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QFrame>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QToolButton>
#include <QButtonGroup>
#include <QCheckBox>
#include <QPalette>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QStackedWidget>
#include <QSlider>
#include <QToolTip>
#include <QtSvgWidgets>
#include <QMessageBox>
#include <QLayout>
#include <QFont>
#include <QDialog>
#include <QTimer>
#include <QProgressBar>

#include <GraphMol/MolDraw2D/MolDraw2DSVG.h>

#include <Search/tree_utils.h>

#define enFont(fontSize) QFont enfont("Calibri", fontSize)

template <typename T>
void setButton(T *button, const str &path, const str &content=""){
    str resourcePath = std::filesystem::current_path().parent_path();
    resourcePath += "/SearchUI/resources/" + path;
    button->setText(content.c_str());
    QIcon icon;
    icon.addFile(resourcePath.c_str(), QSize());
    button->setIcon(icon);
    button->setIconSize(QSize(30, 30));
    button->setCursor(QCursor(Qt::PointingHandCursor));
}