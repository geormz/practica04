import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos desde "irisbin.csv"
data = np.genfromtxt("irisbin.csv", delimiter=",")
X = data[:, :-3]  # Características (dimensiones de pétalos y sépalos)
y = data[:, -3:]  # Etiquetas (código binario de especies)

# Establece el número de divisiones para Leave-k-Out
k = 5  # Puedes ajustar k según tus necesidades

# Inicializa listas para almacenar resultados
accuracy_scores = []
average_accuracy = 0

# Inicializa Leave-One-Out
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Paso 2: Definir y entrenar un modelo de red neuronal
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=10)

    # Paso 3: Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Paso 4: Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred_binary)
    accuracy_scores.append(accuracy)

# Paso 5: Calcula el promedio y desviación estándar de la precisión
average_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

print(f"Promedio de precisión (Leave-One-Out): {average_accuracy}")
print(f"Desviación estándar de precisión (Leave-One-Out): {std_accuracy}")

# Paso 6: Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Paso 7: Visualización en 2D
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(X_reduced[y[:, i] == 1, 0], X_reduced[y[:, i] == 1, 1], c=colors[i], label=f'Species {i}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(loc='best')
plt.title('Visualización de especies Iris en 2D')
plt.show()
