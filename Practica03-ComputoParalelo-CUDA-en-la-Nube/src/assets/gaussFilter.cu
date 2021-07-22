/*
    1. Recibimos como parametros, el canal RGB con el que se va a trabajar
    2. lugar en donde se va almacenar el resultado
    3. ancho de la imagen
    4. alto de la imagen
    5. matriz de gauss
    6. dimension de la matriz de gauss 3 o 5
*/
__global__ void aplicarFiltroGauss(const unsigned char *inputEspacioColor, 
                                    unsigned char *outputEspacioColor, 
                                    const unsigned int ancho, 
                                    const unsigned int alto, 
                                    const float *gausskernel, 
                                    const unsigned int kernel) {
    
    // Obtenemos las columnas resultantes de la multiplicacion del numero
    // de hilos*tamano del bloque *dimension del bloque todas en el espacio de X
    const unsigned int columnas = threadIdx.x + blockIdx.x * blockDim.x;
    // Obtenemos las filas resultantes de la multiplicacion del numero
    // de hilos*tamano del bloque *dimension del bloque todas en el espacio de Y
    const unsigned int filas = threadIdx.y + blockIdx.y * blockDim.y;

    //Comprobacion para ver si no se ha superado las dimensiones de la imagen 
    if(filas < alto && columnas < ancho) {
        // Obtenemos el valor central de la matriz 5//2 = 2.5 = 2 #
        const int mitadSizeKernel = (int)kernel / 2;
        // Variable que van a almacenar la suma de producto de los canales rgb por cada valor de la matriz de gauss
        float pixel = 0.0;
        // Bucles para recorrer la matriz de gauss desde (-2,2]#
        for(int i = -mitadSizeKernel; i <= mitadSizeKernel; i++) {
            for(int j = -mitadSizeKernel; j <= mitadSizeKernel; j++) {

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
               
            }
        }
        //printf("%.0f",pixel);
        // Obtenemos la posicion del pixel haciendo uso de columnas, filas y ancho, luego asignamos el valor del pixel modificado.
        outputEspacioColor[columnas + filas * ancho] = static_cast<unsigned char>(pixel);
    }
}