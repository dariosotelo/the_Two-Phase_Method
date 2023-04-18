#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:07:55 2023

@author: darios
"""

import numpy as np

#Inicializamos una matriz vacía para poder empezar el código
row=int(input("Número de restricciones "))
columns=int(input("Número de variables "))
mat = np.zeros((row, columns+row))

#Llenar la matriz
for r in range(0,row):
    for c in range(0,columns):
        #En esta línea r y c sirven para guiarte en el renglón y columna en el que se encuentra el ciclo
        mat[(r),(c)]=(input("Elemento a["+str(r)+","+str(c)+"] "))

vectorb=np.zeros(row)
for i in range(0, row):
    vectorb[i]=float(input("Pon el valor de la capacidad de la restricción "+str(i+1)+": "))


#Falta poner el caso que no tenga ni variable de holgura ni variable de exceso.
#Ponemos las variables de holgura o de exceso y armamos el vector b
for r in range(0, row):
    hol=int(input("Pon '1' si tu restricción "+str(r+1)+" va a necesitar variable de holgura o '0' si necesita variable de exceso "))
    if (hol==0):
        for c in range(0, columns):
            mat[(r), (c)]=(-1)*mat[(r), (c)]
            vectorb[r]=vectorb[r]*(-1)
    #Estas líneas son las que arman la parte de la matrix con las variables de holgura o de exceso
    for c in range(columns, columns+row):
        if (r+columns==c):
            mat[(r), (c)]=1
        else:
            mat[(r), (c)]=0

#Vamos a armar el vector c
vectorc=np.zeros(columns+row)

for i in range(0, columns+row):
    if (i<columns):
        vectorc[i]=float(input("Pon el valor de los coeficientes de la función de maximización o minimización: "))

#Preguntamos si se quiere maximizar o minimizar el problema
typ=int(input("Pon '1' si quieres maximizar o '0' si quieres minimizar."))


print(vectorb)
print(vectorc)
print(mat)
print(typ)
