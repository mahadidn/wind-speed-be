from rest_framework import serializers
from .models import User, ModelKeras

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'  # Serialize all fields of the User model
        
class ModelKerasSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelKeras
        fields = '__all__'  # Serialize all fields of the ModelKeras model