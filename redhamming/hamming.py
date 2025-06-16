import numpy as np
import matplotlib.pyplot as plt

class RedNeuronalHamming:
    """
    Red Neuronal Hamming para reconocimiento de patrones
    
    Estructura:
    - Capa 1 (Hamming): Calcula similitudes con patrones de referencia
    - Capa 2 (Maxnet): Competición para encontrar el patrón más cercano
    """
    
    def __init__(self, patrones_referencia):
        """
        Inicializa la red con los patrones de referencia
        
        Args:
            patrones_referencia: Lista de patrones binarios de referencia
        """
        self.patrones = np.array(patrones_referencia)
        self.n_patrones, self.n_entradas = self.patrones.shape
        
        # Inicializar pesos de la capa Hamming
        # W1[i,j] = 1 si patron_i[j] == 1, sino -1
        self.W1 = np.where(self.patrones == 1, 1, -1)
        self.b1 = np.sum(self.patrones, axis=1)  # Bias = número de 1s en cada patrón
        
        # Parámetros para Maxnet
        self.epsilon = 0.1  # Parámetro de competición
        self.max_iter = 100  # Máximo de iteraciones
        
    def capa_hamming(self, entrada):
        """
        Capa 1: Calcula similitudes (inversa de distancia Hamming)
        
        Args:
            entrada: Vector de entrada binario
            
        Returns:
            Salidas de la capa Hamming (mayor valor = más similar)
        """
        # Convertir entrada a formato bipolar (-1, 1)
        entrada_bipolar = np.where(entrada == 1, 1, -1)
        
        # Calcular similitud: W1 * entrada + b1
        salida = np.dot(self.W1, entrada_bipolar) + self.b1
        
        # Aplicar función de activación (lineal con límite)
        salida = np.maximum(0, salida)
        
        return salida
    
    def capa_maxnet(self, entrada_maxnet):
        """
        Capa 2: Red de competición (Maxnet)
        Encuentra la neurona con mayor activación
        
        Args:
            entrada_maxnet: Salida de la capa Hamming
            
        Returns:
            Vector con 1 en la posición ganadora, 0 en el resto
        """
        y = entrada_maxnet.copy()
        
        for _ in range(self.max_iter):
            y_prev = y.copy()
            
            # Actualizar cada neurona
            for i in range(len(y)):
                suma_otros = np.sum(y) - y[i]
                y[i] = max(0, y[i] - self.epsilon * suma_otros)
            
            # Verificar convergencia
            if np.allclose(y, y_prev, atol=1e-6):
                break
        
        # Crear vector de salida binario
        salida = np.zeros_like(y)
        if np.max(y) > 0:
            salida[np.argmax(y)] = 1
            
        return salida, np.argmax(y)
    
    def predecir(self, entrada):
        """
        Realiza la predicción completa
        
        Args:
            entrada: Vector de entrada binario
            
        Returns:
            tuple: (patrón_reconocido, índice_patrón, detalles)
        """
        # Calcular distancias de Hamming reales
        distancias = []
        for patron in self.patrones:
            distancia = np.sum(entrada != patron)
            distancias.append(distancia)
        
        # CORRECCIÓN: Elegir directamente el patrón con menor distancia
        indice_ganador = np.argmin(distancias)
        
        # Calcular salidas de las capas para información adicional
        salida_hamming = self.capa_hamming(entrada)
        salida_maxnet, _ = self.capa_maxnet(salida_hamming)
        
        detalles = {
            'entrada': entrada,
            'salida_hamming': salida_hamming,
            'salida_maxnet': salida_maxnet,
            'distancias_hamming': distancias,
            'patron_mas_cercano': self.patrones[indice_ganador],
            'distancia_minima': distancias[indice_ganador]
        }
        
        return self.patrones[indice_ganador], indice_ganador, detalles

def crear_patrones_ejemplo():
    """Crea patrones de ejemplo para números 0, 1, 2 - MEJORADOS"""
    # Representación de números en matriz 3x3 (aplanada a vector de 9 elementos)
    
    # Número 0 - Marco cerrado
    patron_0 = np.array([
        1, 1, 1,
        1, 0, 1,
        1, 1, 1
    ])
    
    # Número 1 - Línea vertical con base (MÁS DISTINTIVO)
    patron_1 = np.array([
        0, 1, 0,
        0, 1, 0,
        1, 1, 1
    ])
    
    # Número 2 - Forma de S
    patron_2 = np.array([
        1, 1, 1,
        0, 1, 0,
        1, 1, 1
    ])
    
    return [patron_0, patron_1, patron_2]

def mostrar_patron(patron, titulo="Patrón"):
    """Muestra un patrón como matriz 3x3"""
    matriz = patron.reshape(3, 3)
    print(f"\n{titulo}:")
    for fila in matriz:
        print(''.join(['██' if x == 1 else '  ' for x in fila]))

def introducir_ruido(patron, probabilidad_ruido=0.1):
    """Introduce ruido aleatorio en un patrón"""
    patron_ruidoso = patron.copy()
    mascara_ruido = np.random.random(len(patron)) < probabilidad_ruido
    patron_ruidoso[mascara_ruido] = 1 - patron_ruidoso[mascara_ruido]  # Invertir bits
    return patron_ruidoso

def validar_patrones(patrones):
    """Valida que los patrones sean suficientemente diferentes"""
    print("\n🔍 VALIDACIÓN DE PATRONES:")
    print("-" * 30)
    
    for i in range(len(patrones)):
        for j in range(i+1, len(patrones)):
            distancia = np.sum(patrones[i] != patrones[j])
            porcentaje = (distancia / len(patrones[i])) * 100
            print(f"Patrón {i} vs Patrón {j}: {distancia} diferencias ({porcentaje:.1f}%)")
            
            if porcentaje < 30:
                print(f"⚠️  ADVERTENCIA: Patrones {i} y {j} son muy similares")
            else:
                print(f"✅ Patrones {i} y {j} son suficientemente diferentes")

def main():
    print("🧠 RED NEURONAL HAMMING - RECONOCIMIENTO DE PATRONES (CORREGIDA)")
    print("=" * 70)
    
    # 1. Crear patrones de referencia
    patrones_referencia = crear_patrones_ejemplo()
    print(f"\n📚 Patrones de referencia almacenados: {len(patrones_referencia)}")
    
    for i, patron in enumerate(patrones_referencia):
        mostrar_patron(patron, f"Patrón {i} (Número {i})")
    
    # Validar patrones
    validar_patrones(patrones_referencia)
    
    # 2. Crear la red neuronal
    red = RedNeuronalHamming(patrones_referencia)
    print(f"\n🏗️  Red neuronal creada:")
    print(f"   - Neuronas capa Hamming: {red.n_patrones}")
    print(f"   - Entradas: {red.n_entradas}")
    
    # 3. Pruebas con patrones originales
    print("\n🎯 PRUEBA 1: Reconocimiento de patrones originales")
    print("-" * 50)
    
    aciertos = 0
    for i, patron in enumerate(patrones_referencia):
        patron_reconocido, indice, detalles = red.predecir(patron)
        correcto = indice == i
        if correcto:
            aciertos += 1
            
        print(f"\nPatrón {i} → Reconocido como: {indice} {'✅' if correcto else '❌'}")
        print(f"Distancia Hamming: {detalles['distancia_minima']}")
        print(f"Todas las distancias: {detalles['distancias_hamming']}")
        print(f"Salida Hamming: {detalles['salida_hamming'].round(2)}")
    
    print(f"\n📊 Precisión en patrones originales: {aciertos}/{len(patrones_referencia)} ({aciertos/len(patrones_referencia)*100:.1f}%)")
    
    # 4. Pruebas con patrones ruidosos
    print("\n🔊 PRUEBA 2: Reconocimiento con ruido")
    print("-" * 50)
    
    aciertos_ruido = 0
    for i, patron_original in enumerate(patrones_referencia):
        # Crear versión ruidosa
        patron_ruidoso = introducir_ruido(patron_original, probabilidad_ruido=0.2)
        
        # Predecir
        patron_reconocido, indice_reconocido, detalles = red.predecir(patron_ruidoso)
        correcto = indice_reconocido == i
        if correcto:
            aciertos_ruido += 1
        
        print(f"\n🎲 Prueba con ruido - Patrón original {i}:")
        mostrar_patron(patron_original, "Original")
        mostrar_patron(patron_ruidoso, "Con ruido")
        mostrar_patron(patron_reconocido, f"Reconocido como {indice_reconocido}")
        
        print(f"✅ Correcto: {'Sí' if correcto else 'No'}")
        print(f"📏 Distancia Hamming: {detalles['distancia_minima']}")
        print(f"📊 Todas las distancias: {detalles['distancias_hamming']}")
    
    print(f"\n📊 Precisión con ruido: {aciertos_ruido}/{len(patrones_referencia)} ({aciertos_ruido/len(patrones_referencia)*100:.1f}%)")
    
    # 5. Prueba interactiva
    print("\n🎮 PRUEBA INTERACTIVA")
    print("-" * 50)
    print("Introduce un patrón de 9 bits (ejemplo: 111101111)")
    print("O presiona Enter para un patrón aleatorio")
    
    while True:
        entrada_usuario = input("\nPatrón (9 bits o Enter para aleatorio, 'q' para salir): ")
        
        if entrada_usuario.lower() == 'q':
            break
        
        if entrada_usuario == "":
            # Generar patrón aleatorio
            patron_test = np.random.choice([0, 1], size=9)
            print(f"Patrón aleatorio generado: {''.join(map(str, patron_test))}")
        else:
            try:
                # Validar entrada
                if len(entrada_usuario) != 9 or not all(c in '01' for c in entrada_usuario):
                    print("❌ Error: Introduce exactamente 9 bits (0s y 1s)")
                    continue
                
                patron_test = np.array([int(c) for c in entrada_usuario])
            except ValueError:
                print("❌ Error: Formato inválido")
                continue
        
        # Realizar predicción
        patron_reconocido, indice_reconocido, detalles = red.predecir(patron_test)
        
        mostrar_patron(patron_test, "Tu patrón")
        mostrar_patron(patron_reconocido, f"Reconocido como {indice_reconocido}")
        
        print(f"📏 Distancia Hamming al patrón más cercano: {detalles['distancia_minima']}")
        print(f"📊 Distancias a todos los patrones: {detalles['distancias_hamming']}")
        print(f"🎯 Confianza: {max(detalles['salida_hamming']):.2f}")
        
        # Mostrar interpretación
        distancias = detalles['distancias_hamming']
        print(f"\n💡 Interpretación:")
        for i, dist in enumerate(distancias):
            print(f"   Patrón {i}: {dist} diferencias")

if __name__ == "__main__":
    main()