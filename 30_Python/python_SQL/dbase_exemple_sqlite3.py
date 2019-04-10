# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:51:37 2016

@author: nabil.belahrach
"""
import numpy as np
import pandas as pd
import sqlite3 
import pandas.io.sql as sql

def create_database():
    conn = sqlite3.connect('cinetique.db')
    cursor = conn.cursor()
    for df in [df23_FB1_2016, df41_FB1_2015, df45_FB1_2015]:
        for name in ["df23_FB1_2016", "df41_FB1_2015", "df45_FB1_2015"]:
            df.to_sql(name= "{:}".format(name), con=conn, if_exists='replace')    
    conn.close()
    return;
    
def read_data():
    conn = sqlite3.connect('cinetique.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type ='table';")
    cursor.execute("SELECT * FROM df23_FB1_2016;")
    print(cursor.fetchall())
    conn.commit()
    read_data = sql.read_sql(" SELECT * FROM df45_FB1_2015 ", conn)
    return; 
#==============================================================================
"""--------------------- main() ---------------------------------------- """ 
#==============================================================================
if __name__ == "__main__":
    #create_database()
    read_data()
    

dfr["semaine"] = 45
dfr = dfr[["nom","semaine", "num_essai","moyenne","ecart-type", "solidite", "entropie", "smoothness"]]


image = ["BLOB (Size: 14949)", "BLOB (Size: 15949)", "BLOB (Size: 1494)", 
          "BLOB (Size: 14149)", "BLOB (Size: 11494)", "BLOB (Size: 11294)",
           "BLOB (Size: 8149)", "BLOB (Size: 17494)", "BLOB (Size: 18295)",
          "BLOB (Size: 14147)", "BLOB (Size: 11464)", "BLOB (Size: 11212)"]
for i in xrange(len(image)):
    dfr.image[i] = image[i]
    
    
    