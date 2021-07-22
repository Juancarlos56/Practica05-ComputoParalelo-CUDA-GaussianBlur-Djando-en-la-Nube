from ConvolucionImagenGPU.logica.GaussFilterCPU import GaussFilterCPU
from django.db import models
from django.urls import reverse
from ConvolucionImagenGPU import models
from itertools import product
from PIL import Image 
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
import numpy as np
import math
import sys
import timeit
import pandas as pd
#import pycuda.autoinit
#import pycuda.driver as drv
#import pycuda.compiler as compiler


class GaussFilterGPU():
    
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
                kernel_matrix[i + centro_del_kernel][j + centro_del_kernel] = GaussFilterGPU.gaussFilter(i,j, sigma)
                #print(" i + centro_del_kernel ", i + centro_del_kernel)
                #print(" j + centro_del_kernel ", j + centro_del_kernel)
                #print("valor = ", gaussFilter(i,j))
        # dividimos la matriz para el resultado. 
        kernel_matrix = kernel_matrix / kernel_matrix.sum()
        return kernel_matrix

    def gaussFilterGPU(path, kernelMascara, desviacionEstandar):
        # Carga de imagen en RGB en la matriz y extraer sus canales de color #
        try:
            # read original image #
            img = Image.open(path)
            # Pasamos la imagen a un array de numpy para obtener los canales #
            img_input_array = np.array(img)
            
            # Creacion de arrays de numpy para cada canal. #
            red = img_input_array[:, :, 0].copy()
            green = img_input_array[:, :, 1].copy()
            blue = img_input_array[:, :, 2].copy()

        except FileNotFoundError:
            sys.exit("No se pudo cargar la imagen")
        
        # Generando gaussian kernel (size of N * N) #1234 35
        kernel = kernelMascara
        sigma = desviacionEstandar

        # LLamada a la funcion para la generacion de matriz de Gauss de NxN #
        gaussian_kernel = GaussFilterGPU.gen_gaussian_kernel(kernel, sigma)
        #gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
        # Calculo de threats/blocks/gird basado en el ancho y altura de una imagen #


        # Obtencion del alto y ancho de una imagen hacemos uso de un canal blue. 
        alto, ancho = img_input_array.shape[:2]

        # Dimension maxima por bloque
        dimension_por_bloque = 32

        # Dimension de cuadrilla para "x" y "y" 
        # ceil nos devuelve un valor entero de la division obtenida.
        dim_grid_x = int(math.ceil(ancho / dimension_por_bloque))
        dim_grid_y = int(math.ceil(alto / dimension_por_bloque))

        # Llamada a funcion de pycuda para obtener respuesta.
        # Leemos la funcion almacenada en el archivo gaussFilter.cu 
        mod = compiler.SourceModule(open('ConvolucionImagenGPU/logica/gaussFilter.cu').read())
        # Obtencion de la funcion de CUDA
        filtroGauss = mod.get_function('aplicarFiltroGauss')

        # Aplicacion de Filtro de Gauss #
        # paso de parametros para la funcion filtroGauss 
        # Toma de tiempo para el programa
        time_started = timeit.default_timer()
        for espacioColor in (red, green, blue):
            # Parametros:
            # 1. Input: canal que pasamos
            # 2. Output: canal que se recupera y se almacena en la misma variable
            # 3. ancho imagen
            # 4. alto imagen
            # 5. Matriz de Gauss
            # 6. Size de kernel
            # 7. block
            # 8. grid
            filtroGauss(
                drv.In(espacioColor),
                drv.Out(espacioColor),
                np.uint32(ancho),
                np.uint32(alto),
                drv.In(gaussian_kernel),
                np.uint32(kernel),
                block=(dimension_por_bloque, dimension_por_bloque, 1),
                grid=(dim_grid_x, dim_grid_y)
            )
        time_ended = timeit.default_timer()
        # display total time
        tiempoFinal = time_ended - time_started
        # Crear la matriz de salida con la misma forma y tipo que la matriz de entrada #
        # Creacion de una matriz de ceros del mismo size de la imagen original
        img_output_array = np.empty_like(img_input_array)
        # Union de cada canal rojo, azul y verde
        #print(red)
        img_output_array[:, :, 0] = red
        img_output_array[:, :, 1] = green
        img_output_array[:, :, 2] = blue
        
        # Guardar imagen Resultados
        pathImgResultado = 'assets/images/resultado.png'
        Image.fromarray(img_output_array).save(pathImgResultado)
        return tiempoFinal, pathImgResultado, dimension_por_bloque, dim_grid_x, dim_grid_y

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
        
        tiempoFinal, pathImgResultado, dimension_por_bloque, dim_grid_x, dim_grid_y = GaussFilterGPU.gaussFilterGPU(path, mascara, desviacion)
        #print("tiempoFinal: ", tiempoFinal)
        
        #resultado = pd.DataFrame([], columns = ['Tiempo' , 'path', 'dimensionbloque', 'gridX', 'gridY'])
        #resultado.loc[0]= [tiempoFinal, pathImgResultado, dimension_por_bloque, dim_grid_x, dim_grid_y]

        #return resultado
        hilos = "X = "+ dim_grid_x + " Y = "+ dim_grid_y
        
        return dimension_por_bloque, hilos, tiempoFinal, mascara, desviacion 
