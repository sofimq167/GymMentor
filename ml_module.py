import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Datos sintéticos de entrenamiento - NUEVAS FEATURES
# Features: [años_experiencia, días_entrena_semana, conoce_tecnica_0_10, 
#            dominadas_consecutivas, sigue_rutina_0_1]

X_train = np.array([
    # Principiantes (recién empezando)
    [0, 2, 2, 0, 0],      # nuevo, 2 días, técnica baja, 0 dominadas, improvisa
    [0, 3, 3, 1, 0],      # nuevo, 3 días, técnica media-baja, 1 dominada
    [0.5, 2, 3, 0, 0],    # 6 meses, 2 días, técnica media-baja
    [0.5, 3, 4, 2, 1],    # 6 meses, 3 días, empieza a seguir rutina
    [1, 3, 4, 3, 1],      # 1 año, pero solo 3 días
    [0, 2, 2, 0, 0],      # nuevo
    [0.3, 3, 3, 1, 0],    # 4 meses
    
    # Intermedios (consistentes, entienden conceptos)
    [1.5, 4, 6, 5, 1],    # 1.5 años, 4 días, buena técnica, 5 dominadas
    [2, 4, 7, 6, 1],      # 2 años, consistente
    [2, 5, 7, 8, 1],      # 2 años, 5 días, buena técnica
    [2.5, 4, 8, 7, 1],    # 2.5 años
    [1.5, 5, 6, 5, 1],    # 1.5 años, muy consistente
    [3, 4, 7, 9, 1],      # 3 años
    [2, 5, 8, 6, 1],      # 2 años, 5 días
    
    # Avanzados (mucha experiencia y conocimiento)
    [3, 5, 9, 12, 1],     # 3 años, 5 días, excelente técnica, 12 dominadas
    [4, 6, 9, 15, 1],     # 4 años, 6 días, excelente
    [5, 5, 10, 18, 1],    # 5 años, muy avanzado
    [3.5, 6, 9, 14, 1],   # 3.5 años, 6 días
    [4, 5, 10, 16, 1],    # 4 años, 5 días
    [5, 6, 10, 20, 1],    # 5+ años, elite
    [6, 6, 10, 20, 1],    # experto
])

y_train = np.array([
    0, 0, 0, 0, 0, 0, 0,      # principiantes
    1, 1, 1, 1, 1, 1, 1,      # intermedios
    2, 2, 2, 2, 2, 2, 2       # avanzados
])

# Entrenar modelo
print("Entrenando clasificador de nivel")
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Calcular accuracy en datos de entrenamiento
accuracy = clf.score(X_train, y_train)
print(f"Accuracy en entrenamiento: {accuracy*100:.1f}%")

# Guardar modelo
with open('nivel_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Modelo guardado como 'nivel_classifier.pkl'")

# Función para predecir
def predecir_nivel(años_exp, dias_semana, conocimiento, dominadas, sigue_rutina):
    """
    Predice el nivel del usuario
    
    Args:
        años_exp: años entrenando (float)
        dias_semana: días que entrena por semana (int)
        conocimiento: conocimiento de técnica 1-10 (int)
        dominadas: cuántas dominadas puede hacer seguidas (int)
        sigue_rutina: 1 si sigue rutina estructurada, 0 si improvisa (int)
    """
    X = np.array([[años_exp, dias_semana, conocimiento, dominadas, sigue_rutina]])
    pred = clf.predict(X)[0]
    niveles = ["Principiante", "Intermedio", "Avanzado"]
    confianza = clf.predict_proba(X)[0][pred]
    return niveles[pred], confianza

# Test
if __name__ == "__main__":
    print("\nCasos de prueba:")
    
    # Caso 1: Principiante
    nivel, conf = predecir_nivel(años_exp=0.5, dias_semana=3, conocimiento=3, 
                                  dominadas=1, sigue_rutina=0)
    print(f"Persona 1 (6 meses, 3 días, técnica=3, 1 dominada, improvisa): {nivel} ({conf*100:.0f}%)")
    
    # Caso 2: Intermedio
    nivel, conf = predecir_nivel(años_exp=2, dias_semana=5, conocimiento=7, 
                                  dominadas=8, sigue_rutina=1)
    print(f"Persona 2 (2 años, 5 días, técnica=7, 8 dominadas, rutina): {nivel} ({conf*100:.0f}%)")
    
    # Caso 3: Avanzado
    nivel, conf = predecir_nivel(años_exp=5, dias_semana=6, conocimiento=10, 
                                  dominadas=20, sigue_rutina=1)
    print(f"Persona 3 (5 años, 6 días, técnica=10, 20 dominadas, rutina): {nivel} ({conf*100:.0f}%)")