# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Work\04_Python\OptiSim V0.5.0_alpha\ui\color.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(820, 536)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.illD65_RB = QtWidgets.QRadioButton(self.groupBox)
        self.illD65_RB.setChecked(True)
        self.illD65_RB.setObjectName("illD65_RB")
        self.verticalLayout.addWidget(self.illD65_RB)
        self.illA_RB = QtWidgets.QRadioButton(self.groupBox)
        self.illA_RB.setObjectName("illA_RB")
        self.verticalLayout.addWidget(self.illA_RB)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.illBB_RB = QtWidgets.QRadioButton(self.groupBox)
        self.illBB_RB.setObjectName("illBB_RB")
        self.horizontalLayout.addWidget(self.illBB_RB)
        self.tempSB = QtWidgets.QSpinBox(self.groupBox)
        self.tempSB.setMaximum(999999)
        self.tempSB.setSingleStep(1)
        self.tempSB.setProperty("value", 3000)
        self.tempSB.setObjectName("tempSB")
        self.horizontalLayout.addWidget(self.tempSB)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.illConst_RB = QtWidgets.QRadioButton(self.groupBox)
        self.illConst_RB.setObjectName("illConst_RB")
        self.verticalLayout.addWidget(self.illConst_RB)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.resultsCB = QtWidgets.QComboBox(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resultsCB.sizePolicy().hasHeightForWidth())
        self.resultsCB.setSizePolicy(sizePolicy)
        self.resultsCB.setMinimumSize(QtCore.QSize(100, 0))
        self.resultsCB.setObjectName("resultsCB")
        self.verticalLayout_4.addWidget(self.resultsCB)
        self.sourceReflection = QtWidgets.QRadioButton(self.groupBox_2)
        self.sourceReflection.setChecked(True)
        self.sourceReflection.setObjectName("sourceReflection")
        self.verticalLayout_4.addWidget(self.sourceReflection)
        self.sourceTransmission = QtWidgets.QRadioButton(self.groupBox_2)
        self.sourceTransmission.setObjectName("sourceTransmission")
        self.verticalLayout_4.addWidget(self.sourceTransmission)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_4.addItem(spacerItem1)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.colorSpecsTE = QtWidgets.QTextEdit(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.colorSpecsTE.sizePolicy().hasHeightForWidth())
        self.colorSpecsTE.setSizePolicy(sizePolicy)
        self.colorSpecsTE.setAutoFormatting(QtWidgets.QTextEdit.AutoNone)
        self.colorSpecsTE.setReadOnly(True)
        self.colorSpecsTE.setObjectName("colorSpecsTE")
        self.verticalLayout_2.addWidget(self.colorSpecsTE)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.colorFrame = QtWidgets.QFrame(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.colorFrame.sizePolicy().hasHeightForWidth())
        self.colorFrame.setSizePolicy(sizePolicy)
        self.colorFrame.setMinimumSize(QtCore.QSize(300, 100))
        self.colorFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.colorFrame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.colorFrame.setLineWidth(2)
        self.colorFrame.setObjectName("colorFrame")
        self.horizontalLayout_3.addWidget(self.colorFrame)
        self.spectrumPlot = MplWidgetSimple(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spectrumPlot.sizePolicy().hasHeightForWidth())
        self.spectrumPlot.setSizePolicy(sizePolicy)
        self.spectrumPlot.setMinimumSize(QtCore.QSize(100, 100))
        self.spectrumPlot.setObjectName("spectrumPlot")
        self.horizontalLayout_3.addWidget(self.spectrumPlot)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setCenterButtons(True)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_3.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Color inspection"))
        self.groupBox.setTitle(_translate("Dialog", "illumnination"))
        self.illD65_RB.setText(_translate("Dialog", "D65 illuminant"))
        self.illA_RB.setText(_translate("Dialog", "illuminant A (black body with T = 2856 K)"))
        self.illBB_RB.setText(_translate("Dialog", "black body illuminant with T =  "))
        self.illConst_RB.setText(_translate("Dialog", "constant illuminant Y = 1.0 over wavelength)"))
        self.groupBox_2.setTitle(_translate("Dialog", "simulation result"))
        self.sourceReflection.setText(_translate("Dialog", "reflection"))
        self.sourceTransmission.setText(_translate("Dialog", "transmission"))
        self.label.setText(_translate("Dialog", "color specifications"))

from ui.mplwidgetsimple import MplWidgetSimple

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

