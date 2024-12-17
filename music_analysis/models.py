from django.db import models

class Tune(models.Model):
    name = models.CharField(max_length=100)  # Tune's name
    composer = models.CharField(max_length=100)  # Composer's name
    abc_notation = models.TextField()  # ABC notation for the tune
    
    def __str__(self):
        return self.name  # String representation of the Tune object
