# Perceptrón desde Cero: ¿Tarjeta Platinum?

Este proyecto muestra cómo funciona un perceptrón para clasificar si una persona obtiene una tarjeta de crédito, según su edad y ahorro.

🔍 **¿Qué hace?** Entrena un perceptrón manualmente (desde cero). Compara resultados con scikit‑learn. Visualiza: Zona de decisión (quién es aprobado o denegado), estructura del perceptrón y frontera de decisión del modelo entrenado.

📦 **Requisitos** Asegúrate de tener Python 3 y estas librerías:
```bash
pip install numpy matplotlib scikit‑learn
▶️ Cómo ejecutar Guarda el archivo como perceptron_tarjeta.py y corre:

bash
Copiar
Editar
python perceptron_tarjeta.py
📈 ¿Qué vas a ver? Pesos y sesgo aprendidos por el perceptrón, comparación con el perceptrón de scikit‑learn, y gráficos que muestran clasificación de personas, zona de decisión, estructura interna del perceptrón y línea de decisión del modelo.

📊 Datos de ejemplo Los datos representan personas con diferentes edades y niveles de ahorro. El modelo aprende a decidir si su solicitud de tarjeta es:

✅ Aprobada
❌ Denegada

------------------------------------------------

# 🧠 Red Neuronal de Hamming para Reconocimiento de Patrones

Este proyecto implementa una **Red Neuronal de Hamming** desde cero en Python, con el objetivo de reconocer patrones binarios (como números 0, 1 y 2 en una matriz de 3x3), incluso si contienen **ruido**.

---

## 📌 ¿Qué hace esta red?

- Compara un patrón de entrada con patrones aprendidos.
- Encuentra el más **similar** usando la distancia de Hamming.
- Usa una red de competición tipo **Maxnet** para decidir cuál patrón gana.
- ¡Funciona incluso con ruido aleatorio!

---

## 🔧 Estructura del código

- `RedNeuronalHamming`: clase principal que contiene dos capas:
  - **Capa Hamming**: mide similitud patrón por patrón.
  - **Capa Maxnet**: decide cuál patrón es el más parecido.
- Patrones de referencia: 0, 1 y 2 en una matriz 3x3.
- Se puede agregar **ruido** y hacer pruebas interactivas.

---

## 🚀 ¿Cómo usarlo?

1. Asegúrate de tener Python 3 y `matplotlib` instalado.
2. Ejecuta el script:


python hamming.py


