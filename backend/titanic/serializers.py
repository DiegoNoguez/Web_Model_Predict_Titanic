from rest_framework import serializers

class PassengerSerializer(serializers.Serializer):
    Name = serializers.CharField(required=False, allow_blank=True)
    Title = serializers.CharField()
    Pclass = serializers.IntegerField(min_value=1, max_value=3)
    Sex = serializers.ChoiceField(choices=['male', 'female'])
    Age = serializers.FloatField(min_value=0, max_value=100)
    SibSp = serializers.IntegerField(min_value=0)
    Parch = serializers.IntegerField(min_value=0)
    Fare = serializers.FloatField(min_value=0)
    Embarked = serializers.CharField()