import pandas as pd
import numpy as np

def preprocess_data(data, features):
    """
    Preprocesa los datos del pasajero para que coincidan con el modelo entrenado
    """
    data = data.copy()
    
    # 1. Procesar título
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Mme', 'Ms']
    data['Title'] = data['Title'].apply(lambda t: t if pd.notna(t) and t in common_titles else 'Other')
    
    # 2. TRANSFORMACIÓN CRÍTICA: Fare a Fare_log
    data['Fare_log'] = np.log1p(data['Fare'])
    
    # 3. Crear características de familia (actualizadas)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone_Negative'] = ((data['SibSp'] + data['Parch']) == 0).astype(int)
    data['HasChildren'] = (data['Parch'] > 0).astype(int)
    data['Single_Male_Penalty'] = (
        (data['Sex'] == 0) & 
        (data['SibSp'] + data['Parch'] == 0) & 
        (data['Age'] > 15)
    ).astype(int)
    data['Mother'] = ((data['Sex'] == 1) & (data['Parch'] > 0) & (data['Age'] > 18)).astype(int)
    
    # 4. Mapear Sex
    sex_mapping = {"male": 0, "female": 1}
    data["Sex"] = data["Sex"].map(sex_mapping).fillna(0).astype(int)
    
    # 5. One-hot encoding
    for col in ['Pclass', 'Embarked', 'Title']:
        data[col] = data[col].astype(str)
    
    data = pd.get_dummies(data, columns=['Pclass', 'Embarked', 'Title'], drop_first=True)
    
    # 6. Asegurar todas las columnas esperadas
    for col in features:
        if col not in data.columns:
            data[col] = 0
    
    # 7. ELIMINAR Fare original - usar solo Fare_log
    if 'Fare' in data.columns:
        data = data.drop('Fare', axis=1)
    
    # 8. Reordenar columnas y asegurar tipos
    data = data[features].astype(float)
    
    return data