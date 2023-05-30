# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:45:18 2023

@author: Carlos Alfredo Pinto Hernández
"""

import os;
import numpy as np;
import cv2 as cv;

rutaProyecto = os.getcwd()
clases = os.listdir(rutaProyecto + '\\BASE_DATOS');
clases_dict = {itemClass: i for i, itemClass in enumerate(clases)}

vectorImagenes=[]
vectorSalida=[]

#Recorrer todas las clases
for clase in clases:
    path= rutaProyecto + '\\BASE_DATOS\\'+clase
    
    #Cargar la imagenes de la clase
    listaImagenes = os.listdir(path)
    #Recorrer todas las imagenes de la clase
    for imagen in listaImagenes:
        #leer cada imagen
        imgOriginal = cv.imread(path + "\\"+imagen, 1)
        if(imgOriginal is None or imgOriginal.shape == (0, 0)):
            print("Error!: " + path + '\\' + imagen)
        else:
            imgFiltered = cv.medianBlur(imgOriginal, 3)
            imgResize = cv.resize(imgFiltered, (128, 128), interpolation = cv.INTER_AREA)
            vectorImagenes.append(imgResize)
            vectorSalida.append(clases_dict[clase])
        
rutaSalida = rutaProyecto + '\\BASE_DATOS_SALIDA'
if(os.path.exists(rutaSalida)):
    print('Archivo actualizado')         
    np.save(rutaSalida + '\\vectorImagenesCNN', vectorImagenes)
    np.save(rutaSalida + '\\vectorYCNN', vectorSalida)
else:
    print('No existe, se crea la carpeta')
    os.mkdir(rutaSalida)
    np.save(rutaSalida + '\\vectorImagenesCNN', vectorImagenes)
    np.save(rutaSalida + '\\vectorYCNN', vectorSalida)
    print('Archivo creado')
        



