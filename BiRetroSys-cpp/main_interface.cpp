#include "main_window.h"

int main(int argc, char *argv[]){
    QApplication app(argc, argv);
    mainWindow w;
    w.loadTerminalMols();
    return app.exec();
}

