from django.db import models

# Create your models here.
class User(models.Model):
    age = models.IntegerField()
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name
    
class ModelKeras(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)