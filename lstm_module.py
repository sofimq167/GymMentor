import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Suprimir warnings de TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Generando datos de entrenamiento...")

# Generar datos sintéticos realistas
def generar_datos(n_samples=2000):
    """
    Genera secuencias de entrenamiento realistas
    1 = entrenó ese día
    0 = no entrenó
    """
    X, y = [], []
    
    for _ in range(n_samples):
        # Simular diferentes tipos de personas
        tipo = np.random.choice(['consistente', 'moderado', 'irregular', 'abandono'])
        
        if tipo == 'consistente':
            # Entrena 5-6 días por semana
            prob = np.random.uniform(0.75, 0.95)
        elif tipo == 'moderado':
            # Entrena 3-4 días por semana
            prob = np.random.uniform(0.50, 0.70)
        elif tipo == 'irregular':
            # Entrena 1-2 días por semana
            prob = np.random.uniform(0.20, 0.45)
        else:  # abandono
            # Casi no entrena
            prob = np.random.uniform(0.05, 0.20)
        
        # Generar secuencia de 7 días
        secuencia = (np.random.rand(7) < prob).astype(float)
        
        X.append(secuencia)
        y.append(prob)  # La probabilidad real de entrenar
    
    return np.array(X), np.array(y)

# Generar datos
X, y = generar_datos(2000)
X = X.reshape(-1, 7, 1)  # (samples, timesteps, features)

print(f"Datos generados: {X.shape[0]} muestras")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Construyendo modelo LSTM")

# Modelo LSTM
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(7, 1), return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("Entrenando LSTM")

# Entrenar
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=0  # Sin output detallado
)

# Evaluar
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Modelo entrenado")
print(f" MAE en test: {mae:.3f} (Error promedio: {mae*100:.1f}%)")

# Guardar
model.save('adherencia_lstm.keras')
print("Modelo guardado como 'adherencia_lstm.keras'")

# Función para predecir
def predecir_adherencia(ultimos_7_dias):
    """
    Predice la probabilidad de entrenar mañana
    
    Args:
        ultimos_7_dias: lista de 7 valores [1,0,1,1,0,1,1]
                       1 = entrenó, 0 = no entrenó
    
    Returns:
        float: probabilidad entre 0 y 1
    """
    X = np.array(ultimos_7_dias).reshape(1, 7, 1)
    return float(model.predict(X, verbose=0)[0][0])

# Casos de prueba
if __name__ == "__main__":
    print("\n Casos de prueba:")
    
    # Caso 1: Muy consistente
    seq1 = [1, 1, 1, 1, 1, 1, 0]
    prob1 = predecir_adherencia(seq1)
    print(f"Secuencia 1 {seq1}: {prob1*100:.1f}% de entrenar mañana")
    
    # Caso 2: Moderado
    seq2 = [1, 0, 1, 1, 0, 1, 0]
    prob2 = predecir_adherencia(seq2)
    print(f"Secuencia 2 {seq2}: {prob2*100:.1f}% de entrenar mañana")
    
    # Caso 3: Abandonando
    seq3 = [0, 0, 1, 0, 0, 0, 0]
    prob3 = predecir_adherencia(seq3)
    print(f"Secuencia 3 {seq3}: {prob3*100:.1f}% de entrenar mañana")
    
    # Caso 4: Recuperándose
    seq4 = [0, 0, 0, 1, 1, 1, 1]
    prob4 = predecir_adherencia(seq4)
    print(f"Secuencia 4 {seq4}: {prob4*100:.1f}% de entrenar mañana")