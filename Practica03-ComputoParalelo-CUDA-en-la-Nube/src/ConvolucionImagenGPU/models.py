from django.db import models

# Create your models here.
# Create your models here.
class ConvolucionGPU(models.Model):
    codigo =models.AutoField(primary_key=True)
    numeroBloques = models.IntegerField() # numero de Bloques GPU 
    numeroHilos = models.IntegerField() # numero de hilos GPU
    tiempoProcesamiento = models.FloatField() # Tiempo de Procesamiento
    mascara = models.IntegerField() # Mascara utilzada
    sigma = models.IntegerField() # Sigma utilizada 
    def __str__(self):
        return str(self.numeroBloques) + ':' + str(self.numeroHilos) + ':' , str(self.tiempoProcesamiento) + ':' , str(self.mascara) + ':'  , str(self.sigma)