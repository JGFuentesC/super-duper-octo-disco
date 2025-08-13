import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Lectura de datos
print("Loading data...")
datos = pd.read_csv('titanic.csv')
print(f"Data shape: {datos.shape}")

# Categorizar variables
varc = ['Age','SibSp','Parch','Fare']  # Variables continuas
vard = ['Embarked','Pclass','Sex']     # Variables discretas
target = 'Survived'

# Preparación de datos - One Hot Encoding
print("Preparing data with One Hot Encoding...")
oh = OneHotEncoder(sparse_output=False, drop='if_binary')
oh.fit(datos[vard])

varoh = list(oh.get_feature_names_out())
print(f"One-hot encoded features: {varoh}")

datos[varoh] = oh.transform(datos[vard])

# Matriz de características X
X = datos[varc+varoh].copy()
print(f"Feature matrix shape: {X.shape}")

# Vector de respuesta y
y = datos[target]

# Imputación de datos
print("Imputing missing values...")
si = SimpleImputer(strategy='median')
si.fit(X)
Xi = si.transform(X)

# Partición de datos
print("Splitting data into train and validation sets...")
X_train, X_valid, y_train, y_valid = train_test_split(Xi, y, test_size=0.3, random_state=42)
y_train.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)

# Entrenamiento con múltiples modelos y selección automática del mejor
print("Training multiple models and selecting the best one...")

# Definir varios modelos para comparar
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Entrenar y evaluar cada modelo
best_model = None
best_score = 0
best_model_name = ""

print("Training and evaluating models...")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Evaluar en validación
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, y_pred_proba)
    
    print(f"{name} - Validation AUC: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with AUC: {best_score:.4f}")

# Asertividad del modelo
print("Evaluating best model performance...")
train_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:,1])
valid_auc = roc_auc_score(y_valid, best_model.predict_proba(X_valid)[:,1])

print(f"Best model ({best_model_name}) performance:")
print(f"Train AUC: {train_auc:.4f}")
print(f"Valid AUC: {valid_auc:.4f}")

# Crear el dataset de score equivalente al final del notebook
print("Creating score dataset...")
aux = pd.DataFrame(best_model.predict_proba(X_valid)[:,1], columns=['Probability'])
aux['real'] = y_valid
aux['muestra'] = 'valid'

auxt = pd.DataFrame(best_model.predict_proba(X_train)[:,1], columns=['Probability'])
auxt['real'] = y_train
auxt['muestra'] = 'train'

score = pd.concat([auxt, aux], ignore_index=True)
score['r_proba'] = pd.cut(score['Probability'], bins=np.arange(0, 1.2, 0.2), include_lowest=True).astype(str)

# Guardar el resultado en CSV
output_file = 'automl_titanic_scores.csv'
score.to_csv(output_file, index=False)
print(f"Score dataset saved to: {output_file}")

# Mostrar resumen del dataset de score
print(f"\nScore dataset shape: {score.shape}")
print("\nFirst few rows of score dataset:")
print(score.head())

print("\nScore distribution by sample:")
print(score['muestra'].value_counts())

print("\nProbability range distribution:")
print(score['r_proba'].value_counts().sort_index())

# Crear gráfico ROC
plt.figure(figsize=(8,6))
ax = plt.gca()
RocCurveDisplay.from_estimator(best_model, X_train, y_train, name="Train", color="blue", ax=ax)
RocCurveDisplay.from_estimator(best_model, X_valid, y_valid, name="Valid", color="orange", ax=ax)

# Baseline (random classifier)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline')

plt.title(f"ROC Curves - {best_model_name} Model (Train vs Valid)")
plt.grid(True)
plt.legend()
plt.savefig('automl_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nROC curve saved as: automl_roc_curves.png")
print(f"Automated ML model training and evaluation completed successfully!")
print(f"Best model selected: {best_model_name}")
