from ConvolucionImagenGPU.logica.GaussFilterCPU import GaussFilterCPU
from django.shortcuts import render
from rest_framework import generics #para microservicio
from ConvolucionImagenGPU import models
from ConvolucionImagenGPU.logica.GaussFilterGPU import GaussFilterGPU
# Create your views here.

class LandingPage():
    def inicio(request):
        #print("Inicioooooooooooooooooooo")
        return render(request, "inicio.html")
    
    def predecir(request):  
        #print("Solicitud de prediccion")
        try:
            #print("Solicitud de prediccion")
            imagen = request.POST.get('imagen')
            mascara = request.POST.get('mascara')
            desviacion = request.POST.get('desviacion')
            
        except:
            imagen = ""
            mascara = ""
            desviacion = ""
        #resul=modeloSNN.modeloSNN.suma(num1,num2)
        
        if imagen != "":
            #resultado = GaussFilterGPU.GaussFilterGPU.predict(imagen, mascara, desviacion)
            bloques, hilos, tiempoFinal, kernel, sigma  = GaussFilterCPU.predict(imagen, int(mascara), int(desviacion))
        else: 
            resultado = "Selecciona la imagen"
        
        return render(request, "inicio.html",{"bloques":bloques , 
                                            "hilos": hilos,
                                            "tiempoFinal": tiempoFinal, 
                                            "kernel": kernel, 
                                            "sigma": sigma})