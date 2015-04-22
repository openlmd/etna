import sqlite3

con = sqlite3.connect('etna.db')
cursor = con.cursor()

#cursor.execute("INSERT INTO test (Id, Name) VALUES (1, 'Pepe')")
#con.commit()
#print 'Guardado correctamente.'

cursor.execute("SELECT Id, Name FROM test")
for reg in cursor:
    print 'ID:', reg[0]
    print 'Name:', reg[1]
    print
    print reg
    
#fd = open('etna.sql', 'r')
#script = fd.read()
#cursor.executescript(script)
#fd.close()
#cursor.commit()

cursor.execute("SELECT * FROM Datos")
for reg in cursor:
    print reg

con.close()
