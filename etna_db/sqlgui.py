import sys
from PySide import QtCore, QtGui, QtSql


class MainForm(QtGui.QWidget):
    def __init__(self):
        super(MainForm, self).__init__()
        #Database:
        self.db = QtSql.QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName("test.db")
        ok = self.db.open()    
        if not ok:
            QtGui.QMessageBox.warning(self, "Error", "Invalid database!")
            return
        # GUI:
        hbox = QtGui.QHBoxLayout(self)
        self.setLayout(hbox)
        
        left_vbox = QtGui.QVBoxLayout()
        right_vbox = QtGui.QVBoxLayout()
        grid = QtGui.QGridLayout()
        grid.setSpacing(10)
        
        self.db_table = QtGui.QTableView(self)
        hbox.addLayout(left_vbox)
        hbox.addLayout(right_vbox) 
        left_vbox.addWidget(self.db_table)
        
        left_vbox.addLayout(grid)
        label_fn = QtGui.QLabel('First Name:')
        label_ln = QtGui.QLabel('Last Name:')
        self.e_fn = QtGui.QLineEdit()
        self.e_ln = QtGui.QLineEdit()
        grid.addWidget(label_fn, 1, 0)
        grid.addWidget(self.e_fn, 1, 1)
        grid.addWidget(label_ln, 2, 0)
        grid.addWidget(self.e_ln, 2, 1)
        
        self.clear_btn = QtGui.QPushButton("Clear")
        self.delete_btn = QtGui.QPushButton("Delete")
        self.refresh_btn = QtGui.QPushButton("Refresh")        
        self.insert_btn = QtGui.QPushButton("Insert")
        self.update_btn = QtGui.QPushButton("Update")
        self.exit_btn = QtGui.QPushButton("Exit")
        right_vbox.addWidget(self.clear_btn)
        right_vbox.addWidget(self.delete_btn)
        right_vbox.addWidget(self.refresh_btn)
        right_vbox.addStretch()
        right_vbox.addWidget(self.insert_btn)
        right_vbox.addWidget(self.update_btn)
        right_vbox.addWidget(self.exit_btn)     
        
        self.setGeometry(300, 100, 800, 600)
        self.setWindowTitle('Simpsons')
        
        #Event Listeners:
        self.refresh_btn.clicked.connect(self.refresh_btn_clicked)
        self.insert_btn.clicked.connect(self.insert_btn_clicked)
        self.delete_btn.clicked.connect(self.delete_btn_clicked) 
        self.clear_btn.clicked.connect(self.clear_btn_clicked) 
        self.update_btn.clicked.connect(self.update_btn_clicked) 
        self.exit_btn.clicked.connect(self.exit_btn_clicked)
        
        #Table:            
        self.model =  QtSql.QSqlQueryModel()
        self.model.setQuery("SELECT * FROM simpsons")
        self.model.setHeaderData(1, QtCore.Qt.Horizontal, self.tr("First Name"))
        self.model.setHeaderData(2, QtCore.Qt.Horizontal, self.tr("Last Name"))
        self.db_table.setModel(self.model) 
        #self.db_table.hideColumn(0) #hide column 'id'
        self.db_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows) #select Row
        self.db_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection) #disable multiselect
        self.show()  
        
    def update_btn_clicked(self):        
        query = QtSql.QSqlQuery()        
        fn = self.e_fn.text()
        ln = self.e_ln.text()
        db_id = self.get_current_id()
        sql = "UPDATE simpsons SET first_name = '%s', last_name = '%s' \
                                  WHERE simpsons_id = '%d'" % (fn, ln, db_id)
        try:
            query.exec_(sql)
            self.db.commit()
        except:
            # Rollback in case there is any error
            self.db.rollback()  
        self.e_fn.clear()
        self.e_ln.clear() 
        self.refresh_table()     
            
    def clear_btn_clicked(self):
        query = QtSql.QSqlQuery()
        query.exec_("DROP TABLE IF EXISTS simpsons")
        sql = """CREATE TABLE simpsons (
                 simpsons_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                 first_name  VARCHAR(20),
                 last_name  VARCHAR(20) )"""  
        query.exec_(sql) 
        self.refresh_table()  
               
        
    def refresh_btn_clicked(self):
        self.refresh_table()       
        
    def insert_btn_clicked(self):
        query = QtSql.QSqlQuery()
        query.prepare("INSERT INTO simpsons (first_name,\
                    last_name) VALUES (:fn, :ln)")
        fn = self.e_fn.text()
        ln = self.e_ln.text()
        self.e_fn.clear()
        self.e_ln.clear()
        query.bindValue(":fn", fn)
        query.bindValue(":ln", ln)
        try:
            query.exec_()
            self.db.commit()
        except:
            self.db.rollback() 
        self.refresh_table()     
        
    def refresh_table(self):        
        self.model.setQuery("SELECT * FROM simpsons")
        
    def delete_btn_clicked(self):
            query = QtSql.QSqlQuery()
            db_id = self.get_current_id()
            sql = "DELETE FROM simpsons WHERE simpsons_id = '%d'" % (db_id)
            try:
                query.exec_(sql)
                self.db.commit()
            except:
                self.db.rollback() 
            self.refresh_table()  
            
    def get_current_id(self):
        if self.db_table.currentIndex():
            #index = self.db_table.selectedIndexes()[0].row() <<--You must use this for multiselect
            index = self.db_table.currentIndex().row()
            db_id = self.model.record(index).value("simpsons_id")
            return db_id
    def exit_btn_clicked(self):
        sys.exit()
     
        
def main():    
    app = QtGui.QApplication(sys.argv)  
    main_form = MainForm()   
    sys.exit(app.exec_())  
        
if __name__ == '__main__':
    main()
    
