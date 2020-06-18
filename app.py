# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FGRRP.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui

import model


# noinspection PyAttributeOutsideInit
class UiMainWindow(object):
    def setup_ui(self, main_window):
        model.run_main_loop()
        main_window.setObjectName("main_window")
        main_window.resize(612, 340)

        self.widget = QtWidgets.QWidget(main_window)
        self.widget.setObjectName("widget")

        self.Logo = QtWidgets.QLabel(main_window)
        self.Logo.setGeometry(QtCore.QRect(232, 185, 150, 140))
        self.Logo.setObjectName("Logo")
        self.Logo.setPixmap(QtGui.QPixmap("./logo.png"))
        self.Logo.setScaledContents(True)
        self.HomeBox = QtWidgets.QGroupBox(self.widget)
        self.HomeBox.setGeometry(QtCore.QRect(40, 30, 171, 121))
        self.HomeBox.setObjectName("HomeBox")
        self.HomeCountry = QtWidgets.QComboBox(self.HomeBox)
        self.HomeCountry.setEnabled(True)
        self.HomeCountry.setGeometry(QtCore.QRect(70, 30, 91, 21))
        self.HomeCountry.setObjectName("HomeCountry")
        self.HomeTeam = QtWidgets.QComboBox(self.HomeBox)
        self.HomeTeam.setGeometry(QtCore.QRect(70, 80, 91, 21))
        self.HomeTeam.setObjectName("HomeTeam")
        self.label = QtWidgets.QLabel(self.HomeBox)
        self.label.setGeometry(QtCore.QRect(10, 30, 58, 15))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.HomeBox)
        self.label_2.setGeometry(QtCore.QRect(10, 80, 58, 15))
        self.label_2.setObjectName("label_2")
        self.AwayBox = QtWidgets.QGroupBox(self.widget)
        self.AwayBox.setGeometry(QtCore.QRect(230, 30, 171, 121))
        self.AwayBox.setObjectName("AwayBox")
        self.AwayCountry = QtWidgets.QComboBox(self.AwayBox)
        self.AwayCountry.setGeometry(QtCore.QRect(70, 30, 91, 21))
        self.AwayCountry.setObjectName("AwayCountry")
        self.AwayTeam = QtWidgets.QComboBox(self.AwayBox)
        self.AwayTeam.setGeometry(QtCore.QRect(70, 80, 91, 21))
        self.AwayTeam.setObjectName("AwayTeam")
        self.label_3 = QtWidgets.QLabel(self.AwayBox)
        self.label_3.setGeometry(QtCore.QRect(10, 30, 58, 15))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.AwayBox)
        self.label_4.setGeometry(QtCore.QRect(10, 80, 58, 15))
        self.label_4.setObjectName("label_4")
        self.ResultBox = QtWidgets.QGroupBox(self.widget)
        self.ResultBox.setGeometry(QtCore.QRect(420, 30, 161, 121))
        self.ResultBox.setObjectName("ResultBox")
        self.resultLabel = QtWidgets.QLabel(self.ResultBox)
        self.resultLabel.setGeometry(QtCore.QRect(10, 50, 141, 16))
        self.resultLabel.setText("")
        self.resultLabel.setObjectName("resultLabel")
        main_window.setCentralWidget(self.widget)
        self.statusbar = QtWidgets.QStatusBar(main_window)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(main_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 612, 25))
        self.menubar.setObjectName("menubar")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        self.menuDisclaimer = QtWidgets.QMenu(self.menubar)
        self.menuDisclaimer.setObjectName("menuDisclaimer")
        main_window.setMenuBar(self.menubar)
        self.actionHelp = QtWidgets.QAction(main_window)
        self.actionHelp.setObjectName("actionHelp")
        self.actionAbout = QtWidgets.QAction(main_window)
        self.actionAbout.setObjectName("actionAbout")
        self.menubar.addAction(self.menuDisclaimer.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        clist = ['Spain', 'Germany', 'France', 'Italy', 'England']
        self.HomeCountry.addItems(clist)
        self.AwayCountry.addItems(clist)
        self.HomeCountry.activated[str].connect(self.home_country_selected)
        self.AwayCountry.activated[str].connect(self.away_country_selected)
        self.AwayTeam.activated[str].connect(self.away_team_selected)

        self.retranslate_ui(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslate_ui(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "Sportify Predictor"))
        self.HomeBox.setTitle(_translate("main_window", "Home"))
        self.label.setText(_translate("main_window", "Country"))
        self.label_2.setText(_translate("main_window", "Team"))
        self.AwayBox.setTitle(_translate("main_window", "Away"))
        self.label_3.setText(_translate("main_window", "Country"))
        self.label_4.setText(_translate("main_window", "Team"))
        self.ResultBox.setTitle(_translate("main_window", "Result"))
        self.menuHelp.setTitle(_translate("main_window", "Help"))
        self.menuAbout.setTitle(_translate("main_window", "About"))
        self.menuDisclaimer.setTitle(_translate("main_window", "File"))
        self.actionHelp.setText(_translate("main_window", "Help"))
        self.actionAbout.setText(_translate("main_window", "About"))

    def home_country_selected(self, text):
        c = model.cnx.cursor()
        c.row_factory = lambda cursor, row: row[0]
        temp = c.execute('SELECT team_short_name AS name FROM Team').fetchall()
        tlist = []
        for i in temp:
            if i not in tlist:
                tlist.append(i)
        self.HomeTeam.addItems(sorted(tlist))

    def away_country_selected(self, text):
        c = model.cnx.cursor()
        c.row_factory = lambda cursor, row: row[0]
        temp = c.execute('SELECT team_short_name AS name FROM Team').fetchall()
        tlist = []
        for i in temp:
            if i not in tlist:
                tlist.append(i)
        self.AwayTeam.addItems(sorted(tlist))

    def away_team_selected(self, text):
        self.predict_result()

    def predict_result(self):
        win = 0
        draw_lose = 0
        t1 = str(self.HomeTeam.currentText())
        t2 = str(self.AwayTeam.currentText())
        t1 = model.short2id.loc[model.short2id['short'] == t1].iloc[0, 0]
        t2 = model.short2id.loc[model.short2id['short'] == t2].iloc[0, 0]
        # 10 Columns for each team: 'B365H', 'B365D', 'B365A', 'BWH', 'BWD',
        #                           'BWA', 'HGA', 'AGA', 'B365', 'BW'

        t1 = model.match.loc[model.match['HomeID'] == t1]
        print(t1)
        if len(t1) > 0:
            testX1 = [t1.iloc[0,6], t1.iloc[0,8]]
        else:
            t1 = model.match.loc[model.match['AwayID'] == t1]
            testX1 = [t1.iloc[0, 7], t1.iloc[0, 9]]

        t2 = model.match.loc[model.match['AwayID'] == t2]
        print(t2)
        if len(t2) > 0:
            testX2 = [t2.iloc[0, 6], t2.iloc[0, 8]]
        else:
            t2 = model.match.loc[model.match['HomeID'] == t2]
            testX2 = [t2.iloc[0, 7], t2.iloc[0, 9]]

        xx = [testX1, testX2]

        # nn_predict = model.dnn.predict(xx)
        # svm_predict = model.cross.predict(xx)
        # nb_predict = model.split.predict(xx)
        #
        # print(nn_predict, svm_predict, nb_predict)
        # if nn_predict == 1:
        #     win += 1
        # else:
        #     draw_lose += 1
        # if svm_predict == 1:
        #     win += 1
        # else:
        #     draw_lose += 1
        # if nb_predict == 1:
        #     win += 1
        # else:
        #     draw_lose += 1
        #
        # if win > draw_lose:
        #     self.resultLabel.setText('Home Win')
        # elif win == draw_lose:
        #     self.resultLabel.setText('Draw')
        # else:
        #     self.resultLabel.setText('Away Win')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UiMainWindow()
    ui.setup_ui(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())
