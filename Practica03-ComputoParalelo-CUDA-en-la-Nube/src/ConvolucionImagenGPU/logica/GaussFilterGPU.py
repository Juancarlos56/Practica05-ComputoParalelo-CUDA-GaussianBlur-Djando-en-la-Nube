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
import pycuda.driver as drv
from pycuda.compiler import SourceModule

MATRIX_SIZE = 1024
BLOCK_SIZE = 1024

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

        drv.init()
        device = drv.Device(0) # enter your gpu id here
        ctx = device.make_context()

        tamano=(MATRIX_SIZE,MATRIX_SIZE)
        
       

        # Dimension de cuadrilla para "x" y "y" 
        # ceil nos devuelve un valor entero de la division obtenida.
        dim_grid_x = int(math.ceil(ancho / dimension_por_bloque))
        dim_grid_y = int(math.ceil(alto / dimension_por_bloque))

        # Llamada a funcion de pycuda para obtener respuesta.
        # Leemos la funcion almacenada en el archivo gaussFilter.cu 
        
        mod = SourceModule(f""" 
        
        __global__ void aplicarFiltroGauss(const unsigned char *inputEspacioColor, 
                                    unsigned char *outputEspacioColor, 
                                    const unsigned int ancho, 
                                    const unsigned int alto, 
                                    const float *gausskernel, 
                                    const unsigned int kernel) {{
    
            // Obtenemos las columnas resultantes de la multiplicacion del numero
            // de hilos*tamano del bloque *dimension del bloque todas en el espacio de X
            const unsigned int columnas = threadIdx.x + blockIdx.x * blockDim.x;
            // Obtenemos las filas resultantes de la multiplicacion del numero
            // de hilos*tamano del bloque *dimension del bloque todas en el espacio de Y
            const unsigned int filas = threadIdx.y + blockIdx.y * blockDim.y;

            //Comprobacion para ver si no se ha superado las dimensiones de la imagen 
            if(filas < alto && columnas < ancho) {{
                // Obtenemos el valor central de la matriz 5//2 = 2.5 = 2 #
                const int mitadSizeKernel = (int)kernel / 2;
                // Variable que van a almacenar la suma de producto de los canales rgb por cada valor de la matriz de gauss
                float pixel = 0.0;
                // Bucles para recorrer la matriz de gauss desde (-2,2]#
                for(int i = -mitadSizeKernel; i <= mitadSizeKernel; i++) {{
                    for(int j = -mitadSizeKernel; j <= mitadSizeKernel; j++) {{

                        // Obtenemos la posicion "x" y "y" de un pixel en especifico para manipularlo. #
                        // Con el min aseguramos no pasarnos del ancho o alto de la imagen #
                        // Con el max aseguramos obtener solo valores positivos y no se problemas con la dimension de la imagen
                        const unsigned int y = max(0, min(alto - 1, filas + i));
                        const unsigned int x = max(0, min(ancho - 1, columnas + j));

                    
                        //Recordamos que la matriz en este caso es una lista no una matriz seguida por lenguaje C
                        //entonces para solo se necesita la posicion una posicion para el kernel que buscamos.
                        const float valorGauss = gausskernel[(j + mitadSizeKernel) + (i + mitadSizeKernel) * kernel];
                        //ahora multiplicamos el valor de la matriz de gauss por el pixel en la posicion x,y 
                        pixel += valorGauss * inputEspacioColor[x + y * ancho];
                    
                    }}
                }}
                //printf("%.0f",pixel);
                // Obtenemos la posicion del pixel haciendo uso de columnas, filas y ancho, luego asignamos el valor del pixel modificado.
                outputEspacioColor[columnas + filas * ancho] = static_cast<unsigned char>(pixel);
            }}
        }}
        
        """)
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
        ctx.pop()
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
        hilos = "X = "+ str(dim_grid_x) + " Y = "+ str(dim_grid_y)
        
        return dimension_por_bloque, hilos, tiempoFinal, mascara, desviacion 
