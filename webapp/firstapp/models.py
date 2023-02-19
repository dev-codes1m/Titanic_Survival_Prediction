from django.db import models

# Create your models here.

class Titanic_Survival(models.Model):
    Pclass = models.FloatField()
    Sex = models.FloatField()
    Age = models.FloatField()
    SibSp = models.FloatField()
    Parch = models.FloatField()
    Fare = models.FloatField()
    Embarked = models.FloatField()
    Fare_Tax = models.FloatField()
    Food_Charges = models.FloatField()
    Luggage_Charges = models.FloatField()
    Survived = models.FloatField()
    