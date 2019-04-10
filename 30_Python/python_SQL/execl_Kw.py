# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:40:54 2016

@author: khalil
"""

import pandas.io.sql as sql
import mysql.connector
import fonctionfp as fp
import fonctionplot as fct
import pandas as pd
import numpy as np
import seaborn 
#import LearningMeth as lm
import fonctions as f
conn = mysql.connector.connect(host='192.168.1.251',port='3306',
                                                database='symao',
                                                   user='symao',
                                                 password='symao')
                                                 
reglage=2974                                         
df=sql.read_sql("select * from spectreH20",conn)
df1=sql.read_sql("select * from referenceMesure",conn)
dfb=sql.read_sql("select * from RefBio",conn)
df=pd.merge(df, df1, on='NumeroM', how='right')
df=df[df.Reglage==str(reglage)]
cols=list(df)
cols.insert(0, cols.pop(cols.index('Classe')))
df = df.ix[:, cols]
df.index=df.NumeroM
df=df.iloc[:,:61]
df["NumeroM"]=df.index
df.to_excel("ouput.xls")
print df
