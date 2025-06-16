import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.metrics import accuracy_score
import matplotlib.patches as patches

# ================================
# 1. Datos simulados: [edad, ahorro]
# ================================
personas = np.array([
    [0.3, 0.4], [0.4, 0.3], [0.3, 0.2], [0.4, 0.1], [0.5, 0.2],
    [0.4, 0.8], [0.6, 0.8], [0.5, 0.6], [0.7, 0.6], [0.8, 0.5]
])

clases = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0: denegada, 1: aprobada

# ================================
# 2. Función de activación y entrenamiento manual
# ================================
def activacion(pesos, x, b):
    z = np.dot(pesos, x)
    return 1 if z + b > 0 else 0

# Inicialización
np.random.seed(42)
pesos = np.random.uniform(-1, 1, size=2)
b = np.random.uniform(-1, 1)
tasa_de_aprendizaje = 0.01
epocas = 100

errores_por_epoca = []

# Entrenamiento manual
for epoca in range(epocas):
    error_total = 0
    for i in range(len(personas)):
        x = personas[i]
        y = clases[i]
        y_hat = activacion(pesos, x, b)
        error = y - y_hat
        error_total += error**2
        pesos += tasa_de_aprendizaje * error * x
        b += tasa_de_aprendizaje * error
    errores_por_epoca.append(error_total)

print("Pesos finales (manual):", pesos)
print("Sesgo final (manual):", b)

# ================================
# 3. Visualización de zonas de decisión (manual)
# ================================
plt.figure(figsize=(7, 6))
plt.title("Perceptrón desde cero: ¿Tarjeta Platinum?", fontsize=16)
plt.scatter(personas[clases == 0][:, 0], personas[clases == 0][:, 1],
            color="red", marker="x", s=120, label="Denegada")
plt.scatter(personas[clases == 1][:, 0], personas[clases == 1][:, 1],
            color="blue", marker="o", s=120, label="Aprobada")

for edad in np.arange(0, 1.01, 0.05):
    for ahorro in np.arange(0, 1.01, 0.05):
        color = activacion(pesos, [edad, ahorro], b)
        plt.scatter(edad, ahorro, color="blue" if color == 1 else "red",
                    alpha=0.15, s=100, marker="s")

plt.xlabel("Edad")
plt.ylabel("Ahorro")
plt.xlim(0, 1.01)
plt.ylim(0, 1.01)
plt.grid(True)
plt.legend()
plt.show()

# ================================
# 4. Gráfico Estructura del Perceptrón (con pesos reales)
# ================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
plt.title("Estructura del Perceptrón (con pesos entrenados)", fontsize=18)

# Entradas
ax.text(0.5, 5, "Edad\nx₁", fontsize=14, ha='center')
ax.text(0.5, 3, "Ahorro\nx₂", fontsize=14, ha='center')

# Flechas desde entradas
ax.annotate('', xy=(2, 5), xytext=(1.1, 5), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate('', xy=(2, 3), xytext=(1.1, 3), arrowprops=dict(arrowstyle="->", lw=2))

# Mostrar valores de pesos reales
ax.text(2.2, 5.2, f"w₁ = {pesos[0]:.2f}", fontsize=12)
ax.text(2.2, 3.2, f"w₂ = {pesos[1]:.2f}", fontsize=12)

# Flechas hacia la suma
ax.annotate('', xy=(4, 4), xytext=(2.5, 5), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate('', xy=(4, 4), xytext=(2.5, 3), arrowprops=dict(arrowstyle="->", lw=2))

# Círculo de la suma
suma_circle = patches.Circle((4, 4), 0.6, fill=False, lw=2)
ax.add_patch(suma_circle)
ax.text(4, 4, "∑", fontsize=20, ha='center', va='center')

# Mostrar valor del sesgo
ax.text(4, 3.1, f"Sesgo b = {b:.2f}", fontsize=12, ha='center')

# Flecha hacia activación
ax.annotate('', xy=(5.5, 4), xytext=(4.6, 4), arrowprops=dict(arrowstyle="->", lw=2))

# Cuadro activación
ax.add_patch(patches.Rectangle((5.5, 3.5), 1.2, 1, fill=False, lw=2))
ax.text(6.1, 4, "f(z)\n> 0 ?", fontsize=12, ha='center', va='center')

# Flecha hacia salida
ax.annotate('', xy=(7.5, 4), xytext=(6.7, 4), arrowprops=dict(arrowstyle="->", lw=2))

# Salida
ax.text(8, 4.3, "Salida", fontsize=14)
ax.text(8, 3.8, "0: Denegada\n1: Aprobada", fontsize=12)

plt.show()

# ================================
# 5. Entrenamiento con Scikit-learn
# ================================
sk_perceptron = SKPerceptron(max_iter=1000, eta0=0.01, random_state=42)
sk_perceptron.fit(personas, clases)
sk_pred = sk_perceptron.predict(personas)

print("\nPesos (scikit-learn):", sk_perceptron.coef_)
print("Sesgo (scikit-learn):", sk_perceptron.intercept_)
print("Precisión:", accuracy_score(clases, sk_pred))

# ================================
# 6. Frontera de decisión scikit-learn
# ================================
plt.figure(figsize=(7, 6))
plt.title("Scikit-learn Perceptrón", fontsize=16)
plt.scatter(personas[clases == 0][:, 0], personas[clases == 0][:, 1],
            color="red", marker="x", s=120, label="Denegada")
plt.scatter(personas[clases == 1][:, 0], personas[clases == 1][:, 1],
            color="blue", marker="o", s=120, label="Aprobada")

# Frontera: w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b)/w2
w = sk_perceptron.coef_[0]
b_sklearn = sk_perceptron.intercept_[0]
x_vals = np.linspace(0, 1, 100)
y_vals = -(w[0] * x_vals + b_sklearn) / w[1]
plt.plot(x_vals, y_vals, 'k--', label="Frontera Scikit-learn")

plt.xlabel("Edad")
plt.ylabel("Ahorro")
plt.xlim(0, 1.01)
plt.ylim(0, 1.01)
plt.grid(True)
plt.legend()
plt.show()
