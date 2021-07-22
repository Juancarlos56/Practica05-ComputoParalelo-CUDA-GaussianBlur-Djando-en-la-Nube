# Practica05-ComputoParalelo-CUDA-GaussianBlur-Djando-en-la-Nube
Desarrollar una aplicación web usando djanjo, flask, web2py o  cualquier otro framework de Python para que permita a través de  un servicio web realizar la convolución de una imagen usando  PyCUDA

### Desarrollo de aplicación web usando djanjo para generar servicio web en donde se realice la convolución de una imagen usando PyCUDA.
#### 1. Desarrollar un algoritmo usando PyCUDA para la convolución de imágenes
```sh
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
```

#### 2. Desarrollar una aplicación web para procesar el algoritmo del punto 1

##### Interfaz gráfica de la aplicación usando Django

- Para la parte grafica revisar archivo inicio.html dentro de templates

![](/Resultados/interfazgrafica.png)

- Para la aplicacion del filtro de gauss revisar archivo views.py y GaussFilterGPU.py

![](/Resultados/predecir-imagen.jpg)

#### 3. Dockerizar la solución del punto 1 y 2

- En este punto se creo un documento de tipo "Dockerfile" que contiene la lista de instrucciones a ser ejecutadas. 

```sh
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
COPY . /app
WORKDIR /app
RUN apt-get -qq update && \
apt-get -qq install build-essential python3.8-dev python3-pip 
RUN rm /usr/bin/python3 && ln -s python3.8 /usr/bin/python3
RUN pip3 install flask && \
pip3 install pycuda==2020.1
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
EXPOSE 8000
CMD python3 manage.py runserver 0.0.0.0:8000 --noreload
```
- Este archivo apunta a requirements.txt, en donde el archivo contiene una lista de las librerias que necesitamos instalar en nuestro contenedor de docker. 

```sh
pillow
Django
djangorestframework
drf-yasg
numpy
pandas
scikit-learn
gunicorn
polls
markdown
django-filter
pyrebase4
```
#### Abrimos una terminal de ubuntu y mandamos a correr los siguientes comandos: 
- Comando para crear contenedor de docker
```sh
sudo docker build --tag practica5barrerajk --network host /home/usuario/Documentos/BarreraJBarreraK-Django/Practica05-ComputoParalelo-CUDA-GaussianBlur-Djando-en-la-Nube-main/Practica03-ComputoParalelo-CUDA-en-la-Nube/src/
```
- Comando para Correr el contenedor de docker
```sh
sudo docker run --gpus all -p 8000:8000 practica5barrerajk
```
- Comando para listar procesos ejecutandose, ver si existe un proceso docker en ejecución.
```sh
sudo docker ps
```
- Para ejecución de docker, finalizar 
```sh
sudo docker stop "identificador proceso"
```
#### 4. Resultados
- Al momento de iniciar nuestro proyecto, ingresamos con el navegador a la direccion: http://127.0.0.1:8000/
- Aqui vemos la primera pagina de inicio.html
- Establecemos los parametros para nuestro filtro de Gauss: Mascara(Kernel), Desviación Estandar(Sigma)
- En la siguiente imagen podemos ver, algunas configuraciones realizadas:

![](/Resultados/interfazgrafica.png)

-  En la siguiente imagen vemos el resultado de aplicar nuestro filto
![](/Resultados/predecir-imagen.jpg)

### 5. Comparacion de Imagenes
##### Imagen Original
![](/Resultados/imagenOriginalSinFiltroRecorte.png)
##### Imagen aplicando el filtro de Gauss
![](/Resultados/resultadoGaussrecorte.png)
