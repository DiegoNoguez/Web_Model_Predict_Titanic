import os
import joblib
import json
import pandas as pd
import numpy as np
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PassengerSerializer
from .utils.preprocessing import preprocess_data

# Cargar recursos una sola vez al importar el módulo
BASE_DIR = settings.BASE_DIR

model_path = os.path.join(BASE_DIR, 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
features_path = os.path.join(BASE_DIR, 'features.json')

# Verificación robusta de carga de modelos
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, "r") as f:
        features = json.load(f)
    print(" Modelos cargados exitosamente")
    print(f" Modelo tipo: {type(model)}")
    print(f" Scaler tipo: {type(scaler)}")
    print(f" Número de features: {len(features)}")
    print(f" ¿Fare_log en features? {'Fare_log' in features}")
except Exception as e:
    print(f" Error cargando modelos: {str(e)}")
    model = None
    scaler = None
    features = []


class RobustPredictView(APIView):
    def post(self, request):
        serializer = PassengerSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        passenger_data = serializer.validated_data
        
        try:
            # Crear DataFrame con validación robusta
            data = pd.DataFrame([{
                "Pclass": int(passenger_data['Pclass']),
                "Sex": str(passenger_data['Sex']),
                "Age": float(passenger_data['Age']),
                "SibSp": int(passenger_data['SibSp']),
                "Parch": int(passenger_data['Parch']),
                "Fare": float(passenger_data['Fare']),
                "Embarked": str(passenger_data['Embarked']),
                "Title": str(passenger_data['Title'])
            }])
            
            print(" DEBUG - Datos recibidos y normalizados:")
            for col in data.columns:
                print(f"  {col}: {data[col].iloc[0]} (tipo: {type(data[col].iloc[0])})")
            
            # Preprocesamiento mejorado
            X_processed = self.enhanced_preprocess_data(data, features)
            
            print(" DEBUG - Después de preprocesamiento mejorado:")
            print(f"Columnas: {X_processed.columns.tolist()}")
            print(f"Shape: {X_processed.shape}")
            
            # Verificar que tenemos todas las features esperadas
            missing_features = set(features) - set(X_processed.columns)
            if missing_features:
                print(f"  Features faltantes: {missing_features}")
                for feature in missing_features:
                    X_processed[feature] = 0
            
            # Asegurar el orden correcto de columnas
            X_processed = X_processed.reindex(columns=features, fill_value=0)
            
            # Escalar y predecir
            X_scaled = scaler.transform(X_processed)
            
            # Predicción con ajuste de probabilidad
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            
            # Aplicar ajustes basados en conocimiento del dominio
            adjusted_probability = self.apply_domain_knowledge_adjustment(
                probability, passenger_data, X_processed.iloc[0]
            )
            
            # Mensaje explicativo
            explanation = self.generate_explanation(
                passenger_data, prediction, adjusted_probability, X_processed.iloc[0]
            )
            
            return Response({
                "name": passenger_data.get('Name', ''),
                "prediction": int(prediction),
                "probability": round(float(adjusted_probability), 3),
                "adjusted_probability": round(float(adjusted_probability), 3),
                "message": explanation,
                "risk_profile": self.assess_risk_profile(passenger_data, X_processed.iloc[0]),
                "debug_info": {
                    "fare_log_value": float(X_processed['Fare_log'].iloc[0]) if 'Fare_log' in X_processed.columns else 0,
                    "is_alone_negative": int(X_processed['IsAlone_Negative'].iloc[0]) if 'IsAlone_Negative' in X_processed.columns else 0,
                    "single_male_penalty": int(X_processed['Single_Male_Penalty'].iloc[0]) if 'Single_Male_Penalty' in X_processed.columns else 0,
                    "family_size": int(X_processed['FamilySize'].iloc[0]) if 'FamilySize' in X_processed.columns else 0,
                }
            })
            
        except Exception as e:
            print(f" ERROR en predicción robusta: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": f"Error en la predicción: {str(e)}"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def enhanced_preprocess_data(self, data, expected_features):
        """Preprocesamiento mejorado con características robustas"""
        data = data.copy()
        
        # 1. Validación y limpieza de datos
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(22.0)
        data['Fare'] = pd.to_numeric(data['Fare'], errors='coerce').fillna(7.0)
        data['SibSp'] = pd.to_numeric(data['SibSp'], errors='coerce').fillna(0)
        data['Parch'] = pd.to_numeric(data['Parch'], errors='coerce').fillna(0)
        
        # 2. Mapeo robusto de Sex
        sex_mapping = {"male": 0, "female": 1, "hombre": 0, "mujer": 1}
        data["Sex"] = data["Sex"].map(sex_mapping).fillna(0).astype(int)
        
        # 3. Características de familia mejoradas
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        data['IsAlone_Negative'] = data['IsAlone']  # Para compatibilidad
        
        # 4. Single_Male_Penalty mejorado
        data['Single_Male_Penalty'] = (
            (data['Sex'] == 0) & 
            (data['IsAlone'] == 1) & 
            (data['Age'] > 15)
        ).astype(int)
        
        # 5. Fare_log con manejo de edge cases
        data['Fare_log'] = np.log1p(data['Fare'])
        
        # 6. Nuevas características robustas
        data['Age_Class_Interaction'] = data['Age'] * data['Pclass']
        data['Fare_Per_Person'] = data['Fare'] / data['FamilySize']
        data['Fare_Per_Person'] = data['Fare_Per_Person'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # 7. Categorización de edad
        data['IsChild'] = (data['Age'] < 12).astype(int)
        data['IsYoungAdult'] = ((data['Age'] >= 12) & (data['Age'] <= 25)).astype(int)
        data['IsElderly'] = (data['Age'] > 60).astype(int)
        
        # 8. Procesamiento de título
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Mme', 'Ms']
        data['Title'] = data['Title'].apply(lambda t: t if t in common_titles else 'Other')
        
        # 9. One-hot encoding robusto
        data = pd.get_dummies(data, columns=['Pclass', 'Embarked', 'Title'], drop_first=True)
        
        # 10. Asegurar todas las features esperadas
        for feature in expected_features:
            if feature not in data.columns:
                data[feature] = 0
                
        # Eliminar columnas no esperadas
        extra_cols = set(data.columns) - set(expected_features)
        if extra_cols:
            data = data.drop(columns=list(extra_cols))
        
        return data[expected_features]
    
    def apply_domain_knowledge_adjustment(self, probability, passenger_data, processed_features):
        """Ajusta la probabilidad basándose en conocimiento del dominio del Titanic"""
        adjusted_prob = probability
        adjustment_factors = []
        
        # Factores de riesgo históricos
        if passenger_data['Sex'] == 'male':
            adjusted_prob *= 0.6  # Hombres tenían mucha menor supervivencia
            adjustment_factors.append("Ser hombre reduce probabilidad")
            
        if passenger_data['Pclass'] == 3:
            adjusted_prob *= 0.7  # Tercera clase tenía menor supervivencia
            adjustment_factors.append("Tercera clase reduce probabilidad")
            
        if processed_features.get('IsAlone', 0) == 1:
            adjusted_prob *= 0.8  # Viajar solo reducía supervivencia
            adjustment_factors.append("Viajar solo reduce probabilidad")
            
        if passenger_data['Age'] > 15 and passenger_data['Age'] < 60:
            adjusted_prob *= 0.9  # Hombres adultos tenían prioridad baja
            if passenger_data['Sex'] == 'male':
                adjustment_factors.append("Edad adulta reduce probabilidad para hombres")
        
        # Factores positivos
        if passenger_data['Sex'] == 'female':
            adjusted_prob *= 1.8  # Mujeres tenían alta supervivencia
            adjustment_factors.append("Ser mujer aumenta probabilidad")
            
        if passenger_data['Pclass'] == 1:
            adjusted_prob *= 1.4  # Primera clase tenía alta supervivencia
            adjustment_factors.append("Primera clase aumenta probabilidad")
            
        if processed_features.get('IsChild', 0) == 1:
            adjusted_prob *= 1.3  # Niños tenían mayor supervivencia
            adjustment_factors.append("Ser niño aumenta probabilidad")
        
        print(f" Ajustes aplicados: {adjustment_factors}")
        print(f" Probabilidad original: {probability:.3f}, Ajustada: {adjusted_prob:.3f}")
        
        return max(0.05, min(0.95, adjusted_prob))  # Mantener en rango razonable
    
    def assess_risk_profile(self, passenger_data, processed_features):
        """Evalúa el perfil de riesgo del pasajero"""
        risk_score = 0
        factors = []
        
        # Factores de alto riesgo
        if passenger_data['Sex'] == 'male':
            risk_score += 3
            factors.append("Hombre adulto")
            
        if passenger_data['Pclass'] == 3:
            risk_score += 2
            factors.append("Tercera clase")
            
        if processed_features.get('IsAlone', 0) == 1:
            risk_score += 1
            factors.append("Viajando solo")
            
        if passenger_data['Fare'] < 10:
            risk_score += 1
            factors.append("Tarifa baja")
            
        # Factores de bajo riesgo
        if passenger_data['Sex'] == 'female':
            risk_score -= 3
            factors.append("Mujer")
            
        if passenger_data['Pclass'] == 1:
            risk_score -= 2
            factors.append("Primera clase")
            
        if processed_features.get('IsChild', 0) == 1:
            risk_score -= 2
            factors.append("Niño")
        
        if risk_score >= 4:
            profile = "Alto riesgo"
        elif risk_score >= 2:
            profile = "Riesgo moderado"
        elif risk_score >= 0:
            profile = "Riesgo bajo"
        else:
            profile = "Bajo riesgo (favorable)"
            
        return {
            "risk_level": profile,
            "risk_score": risk_score,
            "factors": factors
        }
    
    def generate_explanation(self, passenger_data, prediction, probability, processed_features):
        """Genera un mensaje explicativo de la predicción"""
        name = passenger_data.get('Name', 'El pasajero')
        
        base_message = f"{name} {'SOBREVIVIÓ' if prediction == 1 else 'NO SOBREVIVIÓ'}"
        prob_message = f"Probabilidad: {probability*100:.1f}%"
        
        # Factores clave que influyeron
        key_factors = []
        
        if passenger_data['Sex'] == 'male':
            key_factors.append("ser hombre")
        else:
            key_factors.append("ser mujer")
            
        if passenger_data['Pclass'] == 1:
            key_factors.append("viajar en primera clase")
        elif passenger_data['Pclass'] == 3:
            key_factors.append("viajar en tercera clase")
            
        if processed_features.get('IsAlone', 0) == 1:
            key_factors.append("viajar solo")
        else:
            key_factors.append("viajar con familia")
            
        factors_text = ", ".join(key_factors)
        
        explanation = f"{base_message} ({prob_message}). Factores clave: {factors_text}."
        
        return explanation


# Mantener las vistas existentes para compatibilidad
class PredictView(APIView):
    def post(self, request):
        # Redirigir a la vista robusta
        robust_view = RobustPredictView()
        return robust_view.post(request)


class HealthCheck(APIView):
    def get(self, request):
        return Response({
            "message": "API Titanic activa ", 
            "model_loaded": model is not None,
            "features_count": len(features) if features else 0
        })


class DebugPredictionView(APIView):
    def post(self, request):
        """Endpoint para debug mejorado"""
        serializer = PassengerSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
            
        passenger_data = serializer.validated_data
        
        try:
            # Usar el preprocesamiento robusto
            data = pd.DataFrame([{
                "Pclass": passenger_data['Pclass'],
                "Sex": passenger_data['Sex'],
                "Age": passenger_data['Age'],
                "SibSp": passenger_data['SibSp'],
                "Parch": passenger_data['Parch'],
                "Fare": passenger_data['Fare'],
                "Embarked": passenger_data['Embarked'],
                "Title": passenger_data['Title']
            }])
            
            robust_view = RobustPredictView()
            X_processed = robust_view.enhanced_preprocess_data(data, features)
            X_scaled = scaler.transform(X_processed)
            
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            adjusted_probability = robust_view.apply_domain_knowledge_adjustment(
                probability, passenger_data, X_processed.iloc[0]
            )
            
            # Análisis de contribución de features
            if hasattr(model, 'coef_'):
                contributions = model.coef_[0] * X_scaled[0]
                contrib_df = pd.DataFrame({
                    'feature': features,
                    'contribution': contributions,
                    'abs_contribution': np.abs(contributions)
                }).sort_values('abs_contribution', ascending=False)
                
                top_positive = contrib_df.head(5).to_dict('records')
                top_negative = contrib_df[contrib_df['contribution'] < 0].head(5).to_dict('records')
            else:
                top_positive = []
                top_negative = []
            
            return Response({
                "prediction": int(prediction),
                "probability": round(float(probability), 4),
                "adjusted_probability": round(float(adjusted_probability), 4),
                "risk_profile": robust_view.assess_risk_profile(passenger_data, X_processed.iloc[0]),
                "debug_info": {
                    "fare_original": passenger_data['Fare'],
                    "fare_log": float(X_processed['Fare_log'].iloc[0]) if 'Fare_log' in X_processed.columns else 0,
                    "is_alone": int(X_processed['IsAlone_Negative'].iloc[0]) if 'IsAlone_Negative' in X_processed.columns else 0,
                    "single_male_penalty": int(X_processed['Single_Male_Penalty'].iloc[0]) if 'Single_Male_Penalty' in X_processed.columns else 0,
                    "family_size": int(X_processed['FamilySize'].iloc[0]) if 'FamilySize' in X_processed.columns else 0,
                    "top_positive_features": top_positive,
                    "top_negative_features": top_negative,
                    "features_used": len(X_processed.columns)
                }
            })
            
        except Exception as e:
            print(f"ERROR en debug: {str(e)}")
            return Response({"error": str(e)}, status=500)


# Verificación al cargar el módulo
print("=== VERIFICACIÓN ROBUSTA DE CARGA EN API ===")
print(f" Modelo cargado: {model is not None}")
print(f" Scaler cargado: {scaler is not None}")
print(f" Features cargadas: {len(features) if features else 0}")
print(f" Features críticas presentes: {'Fare_log' in features}, {'IsAlone_Negative' in features}, {'Single_Male_Penalty' in features}")
print(f" Tipo de modelo: {type(model) if model else 'No cargado'}")