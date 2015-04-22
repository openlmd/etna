import sys

from PySide import QtCore, QtGui, QtSql

app = QtGui.QApplication(sys.argv)

db = QtSql.QSqlDatabase.addDatabase("QSQLITE")
db.setDatabaseName("test.db")
db.open()

projectModel = QtSql.QSqlQueryModel()
projectModel.setQuery("SELECT * FROM simpsons",db)

projectView = QtGui.QTableView()
projectView.setModel(projectModel)

projectView.show()
app.exec_()