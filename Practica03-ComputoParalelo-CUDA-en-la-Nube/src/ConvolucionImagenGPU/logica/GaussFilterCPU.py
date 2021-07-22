from django.db import models
from django.urls import reverse
from ConvolucionImagenGPU import models
from itertools import product
from PIL import Image 
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
import numpy as np
import pandas as pd
import math
import sys
import timeit
import signal

class GaussFilterCPU():

    def gaussFilter(x,y, sigma):
        #formula: w(x,y) =  e^(-(x^2+ y^2)/(sigma^2))/(2*pi*sigma^2)
        return ( np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2))
    
    
    #Crear una matriz de numpy tipo float32
    def gen_gaussian_kernel(kernel , sigma):
        # Creacion de una matriz vacia de NxN de tipo float#
        kernel_matrix = np.empty((kernel, kernel), np.float32)
        # Obtenemos el valor central de la matriz 5//2 = 2.5 = 2
        centro_del_kernel = kernel // 2

        # Iteramos desde -2 hasta +3 teniendo 5 posiciones si el kernel es de 5
        for i in range(-centro_del_kernel, centro_del_kernel + 1):
            # Iteramos desde -2 hasta +3 teniendo 5 posiciones si el kernel es de 5
            for j in range(-centro_del_kernel, centro_del_kernel + 1):
                # creamos la matriz en la posicion i,j, va desde 0 a 4 cuando el kernel es 5
                kernel_matrix[i + centro_del_kernel][j + centro_del_kernel] = GaussFilterCPU.gaussFilter(i,j, sigma)
                #print(" i + centro_del_kernel ", i + centro_del_kernel)
                #print(" j + centro_del_kernel ", j + centro_del_kernel)
                #print("valor = ", gaussFilter(i,j))
        #print("***************Matriz Resultado***************")
        #print("Divisor comun para la matriz: ", kernel_matrix.sum())
        # dividimos la matriz para el resultado. 
        kernel_matrix = kernel_matrix / kernel_matrix.sum()
        #print(kernel_matrix)
        return kernel_matrix

    def filtroGauss(img_input_array, sigma, kernel, gauss_matriz):
        # Crear la matriz de salida con la misma forma y tipo que la matriz de entrada #
        # Creacion de una matriz de ceros del mismo size de la imagen original #
        result_array = np.empty_like(img_input_array)
        # Obtencion del alto y ancho de una imagen hacemos uso de un canal blue. #
        alto, ancho = img_input_array.shape[:2]
        # Obtenemos el valor central de la matriz 5//2 = 2.5 = 2 #
        centro_del_kernel = kernel // 2
        
        # Recorrido de cada pixel de la imagen de manera secuencial
        # i va a obtener los valores del alto de la imagen y j los valores del ancho 
        for i in range(0, alto):
            for j in range(0, ancho):
                # Variables que van a almacenar la suma de producto de los canales rgb por cada valor de la matriz de gauss
                red = 0.0
                green = 0.0
                blue = 0.0
                
                # Bucles para recorrer la matriz de gauss#
                for k in range(-centro_del_kernel, centro_del_kernel + 1):
                    for l in range(-centro_del_kernel, centro_del_kernel + 1):

                        # Obtenemos la posicion "x" y "y" de un pixel en especifico para manipularlo. #
                        # Con el min aseguramos no pasarnos del ancho o alto de la imagen #
                        # Con el max aseguramos obtener solo valores positivos y no se problemas con la dimension de la imagen                    
                        x = max(0, min(img_input_array.shape[1] - 1, j + l))
                        y = max(0, min(img_input_array.shape[0] - 1, i + k))
                        #print("-----------------------------------")
                        #print(i," | ",j," | ",x ," | ", y, " | ", img_input_array.shape[1], " | ", img_input_array.shape[1], " | ", img_input_array[y][x])
                        
                        # Obtenemos la posicion del pixel: img_input_array[y][x] = [0,72,166]
                        # Obtenemos los valores de la matriz de Gauss: gauss_matriz[0][0] = 0.0032....
                        # Multiplicamos cada canal rgb por el valores de la matriz de gauss y los almacenamos 
                        r, g, b = (img_input_array[y][x] * gauss_matriz[k + centro_del_kernel][l + centro_del_kernel])
                        # Se suma y almacera los resultados para llegar a obtener los valores reales para nuestro pixel #
                        red += r
                        green += g
                        blue += b
                # Colocamos nuestro nuevo valor para el pixel con gauss aplicado, en los tres espacios de color.#
                result_array[i][j] = (red, green, blue)
        return result_array


    def gaussFilterCPU(path, kernelMascara, desviacionEstandar):
        # Carga de imagen en RGB en la matriz y extraer sus canales de color #
        try:
            # read original image #
            img = Image.open(path)
            # Pasamos la imagen a un array de numpy para obtener los canales #
            img_input_array = np.array(img)

        except FileNotFoundError:
            sys.exit("No se pudo cargar la imagen")
        
        # Generando gaussian kernel (size of N * N) #1234 35
        kernel = kernelMascara
        sigma = desviacionEstandar
        
        # Crear la matriz de salida con la misma forma y tipo que la matriz de entrada #
        # Creacion de una matriz de ceros del mismo size de la imagen original #
        img_output_array = np.empty_like(img_input_array)
        
        #LLamada a la funcion para la generacion de matriz de Gauss de NxN #
        gauss_matriz = GaussFilterCPU.gen_gaussian_kernel(kernel, sigma)
        
        # Aplicacion de Filtro de Gauss #
        # Toma de tiempo para el programa
        time_started = timeit.default_timer()
        # LLamda a la funcion de Gauss para su respectivo calculo#
        img_output_array = GaussFilterCPU.filtroGauss(img_input_array, sigma, kernel, gauss_matriz)
        time_ended = timeit.default_timer()
        # mostrar tiempo total
        tiempoFinal = time_ended - time_started
        

        # Union de cada canal rojo, azul y verde
        #print(red)
        
        # Guardar imagen Resultados
        pathImgResultado = 'assets/images/resultado.png'
        Image.fromarray(img_output_array).save(pathImgResultado)
        return tiempoFinal, pathImgResultado


    def predict(imagen, mascara, desviacion):
        
        if( imagen == "Imagen1"):
            path = "ConvolucionImagenGPU/logica/images/ave.jpg"
        elif(imagen == "Imagen2"):
            path = "ConvolucionImagenGPU/logica/images/lago.jpg"
        elif(imagen == "Imagen3"):
            path = "ConvolucionImagenGPU/logica/images/paisaje-montanas.jpg"
        elif(imagen == "Imagen4"):
            path = "ConvolucionImagenGPU/logica/images/paisaje-nubes.jpg"
        elif(imagen == "Imagen5"):
            path = "ConvolucionImagenGPU/logica/images/paisaje.jpg"
        elif(imagen == "Imagen6"):
            path = "ConvolucionImagenGPU/logica/images/rombo.jpg"
        
        tiempoFinal, pathImgResultado = GaussFilterCPU.gaussFilterCPU(path, mascara, desviacion)
        #print("tiempoFinal: ", tiempoFinal)
        #resultado = pd.DataFrame([], columns = ['Tiempo' , 'path'])
        #resultado.loc[0]= [tiempoFinal, pathImgResultado]
        return 0, 0, tiempoFinal, mascara, desviacion 
        #return 0, 0, 0.1, mascara, desviacion 
