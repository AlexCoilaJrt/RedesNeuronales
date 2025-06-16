import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow version:", tf.__version__)

# Cargar el dataset MNIST
print("Cargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Forma de datos de entrenamiento: {x_train.shape}")
print(f"Forma de etiquetas de entrenamiento: {y_train.shape}")
print(f"Forma de datos de prueba: {x_test.shape}")
print(f"Forma de etiquetas de prueba: {y_test.shape}")

# Visualizar algunas imágenes del dataset
plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Etiqueta: {y_train[i]}')
    plt.axis('off')
plt.suptitle('Ejemplos del dataset MNIST')
plt.tight_layout()
plt.show()

# Normalizar los datos (de 0-255 a 0-1)
print("Normalizando datos...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Aplanar las imágenes (de 28x28 a 784)
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)

print(f"Forma después de aplanar: {x_train_flat.shape}")

# Crear el modelo
print("Creando modelo...")
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,), name='capa_oculta'),
    layers.Dropout(0.2, name='dropout'),
    layers.Dense(10, activation='softmax', name='capa_salida')
])

# Mostrar resumen del modelo
model.summary()

# Visualizar la arquitectura de la red neuronal
def visualizar_arquitectura_red():
    """Función para visualizar la arquitectura de la red neuronal"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Definir posiciones de las capas
    input_layer_size = 5  # Representamos solo 5 neuronas de entrada para simplificar
    hidden_layer_size = 4  # Representamos 4 neuronas ocultas
    output_layer_size = 3  # Representamos 3 neuronas de salida
    
    # Posiciones de las neuronas
    layer_positions = {
        'input': [(0, i) for i in range(input_layer_size)],
        'hidden': [(2, i) for i in range(hidden_layer_size)],
        'output': [(4, 1)]  # Solo una neurona de salida centrada
    }
    
    # Colores para cada capa
    colors = {
        'input': '#FFE5B4',    # Amarillo claro
        'hidden': '#ADD8E6',   # Azul claro
        'output': '#FFB6C1'    # Rosa claro
    }
    
    # Dibujar neuronas
    for layer_name, positions in layer_positions.items():
        for pos in positions:
            circle = plt.Circle(pos, 0.3, color=colors[layer_name], 
                              ec='black', linewidth=2, zorder=3)
            ax.add_patch(circle)
    
    # Dibujar conexiones
    for input_pos in layer_positions['input']:
        for hidden_pos in layer_positions['hidden']:
            ax.plot([input_pos[0], hidden_pos[0]], 
                   [input_pos[1], hidden_pos[1]], 
                   'k-', alpha=0.3, linewidth=0.5)
    
    for hidden_pos in layer_positions['hidden']:
        for output_pos in layer_positions['output']:
            ax.plot([hidden_pos[0], output_pos[0]], 
                   [hidden_pos[1], output_pos[1]], 
                   'k-', alpha=0.3, linewidth=0.5)
    
    # Etiquetas de las capas
    ax.text(0, -1, 'Entrada\n(784 píxeles)', ha='center', va='top', 
            fontsize=12, fontweight='bold')
    ax.text(2, -1, 'Capa Oculta\n(128 neuronas)', ha='center', va='top', 
            fontsize=12, fontweight='bold')
    ax.text(4, -1, 'Salida\n(10 dígitos)', ha='center', va='top', 
            fontsize=12, fontweight='bold')
    
    # Añadir información adicional
    ax.text(1, 3, 'Activación:\nReLU', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    ax.text(3, 3, 'Activación:\nSoftmax', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightpink'))
    ax.text(1, -0.5, 'Dropout: 20%', ha='center', va='center', 
            fontsize=9, style='italic')
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Arquitectura de la Red Neuronal - MNIST', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

# Mostrar la arquitectura antes del entrenamiento
print("\n" + "="*60)
print("VISUALIZACIÓN DE LA ARQUITECTURA DE LA RED NEURONAL")
print("="*60)
visualizar_arquitectura_red()

print("\n" + "="*60)
print("EXPLICACIÓN DEL FUNCIONAMIENTO")
print("="*60)
print("""
🧠 CÓMO FUNCIONA ESTA RED NEURONAL:

1. ENTRADA (Input Layer):
   • Recibe 784 píxeles (28x28 imagen aplanada)
   • Cada píxel tiene un valor entre 0 y 1 (normalizado)

2. CAPA OCULTA (Hidden Layer):
   • 128 neuronas con activación ReLU
   • ReLU: f(x) = max(0, x) - elimina valores negativos
   • Cada neurona recibe: suma ponderada de entradas + bias
   • Dropout del 20%: apaga aleatoriamente neuronas para evitar overfitting

3. CAPA DE SALIDA (Output Layer):
   • 10 neuronas (una para cada dígito 0-9)
   • Activación Softmax: convierte salidas en probabilidades
   • La neurona con mayor probabilidad es la predicción

4. PROCESO DE ENTRENAMIENTO:
   • Forward pass: datos fluyen de entrada a salida
   • Cálculo de error: diferencia entre predicción y etiqueta real
   • Backpropagation: ajusta pesos para minimizar error
   • Optimizador Adam: ajusta automáticamente la tasa de aprendizaje
""")
print("="*60)

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO...")
print("="*60)
print("""
🚀 PROCESO DE ENTRENAMIENTO:
• Épocas: 5 (el modelo verá todos los datos 5 veces)
• Batch size: 128 (procesa 128 imágenes a la vez)
• Validación: 10% de datos reservados para validar
• Optimizador: Adam (ajusta automáticamente learning rate)
• Función de pérdida: Sparse Categorical Crossentropy
""")

history = model.fit(
    x_train_flat, y_train,
    epochs=10,  # Puedes aumentar a 20 o más para mejor precisión
    batch_size=128,
    validation_split=0.1,  # Usar 10% de datos para validación
    verbose=1
)

# Evaluar el modelo
print("\n" + "="*60)
print("EVALUANDO MODELO...")
print("="*60)
test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
print(f'📊 Precisión en test: {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'📊 Pérdida en test: {test_loss:.4f}')

if test_acc > 0.97:
    print("🎉 ¡Excelente! El modelo tiene muy buena precisión")
elif test_acc > 0.95:
    print("👍 Buen rendimiento del modelo")
else:
    print("⚠️  El modelo podría mejorarse")

# Graficar el historial de entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()

# Hacer predicciones en algunas imágenes de prueba
print("Haciendo predicciones...")
predictions = model.predict(x_test_flat[:10])
predicted_classes = np.argmax(predictions, axis=1)

# Visualizar predicciones
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Real: {y_test[i]}, Predicción: {predicted_classes[i]}')
    plt.axis('off')
plt.suptitle('Predicciones del modelo')
plt.tight_layout()
plt.show()

# Mostrar matriz de confusión (opcional)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Predicciones para todo el conjunto de test
all_predictions = model.predict(x_test_flat)
all_predicted_classes = np.argmax(all_predictions, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_test, all_predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.show()

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, all_predicted_classes))

print("\n¡Entrenamiento completado!")
print(f"🎯 El modelo alcanzó una precisión de {test_acc*100:.2f}% en el conjunto de prueba.")
print("\n" + "="*60)
print("RESUMEN DEL PROCESO")
print("="*60)
print(f"""
✅ RESULTADOS FINALES:
• Precisión de entrenamiento: {history.history['accuracy'][-1]*100:.2f}%
• Precisión de validación: {history.history['val_accuracy'][-1]*100:.2f}%
• Precisión de test: {test_acc*100:.2f}%
• Pérdida final: {test_loss:.4f}

🔍 ANÁLISIS:
• La red neuronal aprendió a reconocer dígitos manuscritos
• Cada imagen de 28x28 píxeles se convierte en un vector de 784 números
• La capa oculta de 128 neuronas extrae características importantes
• La capa de salida de 10 neuronas predice qué dígito es (0-9)
• Dropout ayuda a evitar memorización excesiva (overfitting)

🚀 FUNCIONAMIENTO EN TIEMPO REAL:
1. Se toma una imagen de 28x28 píxeles
2. Se normaliza (valores 0-1) y se aplana (vector 784)
3. Se multiplica por pesos aprendidos + bias
4. Se aplica ReLU en capa oculta
5. Se aplica Softmax en salida para obtener probabilidades
6. Se selecciona el dígito con mayor probabilidad
""")