import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, accuracy_score

# Definir caminho das imagens e tamanho
path = 'IMAGES'
img_size = (128, 128)

# Definir os nomes das pastas de treinamento
classes = ['OVELHA', 'VACA']

# Criar lista vazia para armazenar as imagens e os labels
data = []
labels = []

# Loop através das pastas de treinamento
for i, classe in enumerate(classes):
    # Definir o caminho completo da pasta atual
    class_path = os.path.join(path, classe)
    
    # Loop através de cada imagem na pasta atual
    for img in os.listdir(class_path):
        # Definir o caminho completo da imagem atual
        img_path = os.path.join(class_path, img)
        
        # Ler a imagem usando OpenCV
        image = cv2.imread(img_path)
        
        # Verificar se a imagem foi lida corretamente
        if image is None:
            print(f"Erro ao ler a imagem: {img_path}")
            continue
        
        # Redimensionar a imagem para o tamanho especificado
        if image.shape[:2] != img_size:
            image = cv2.resize(image, img_size)
        
        # Adicionar a imagem e o label às listas
        data.append(image)
        labels.append(i)

# Verificar se a lista de dados não está vazia
if not data:
    print("Nenhuma imagem encontrada nas pastas de treinamento!")
    exit()

# Converter as listas em arrays NumPy
data = np.array(data)
labels = np.array(labels)

# Normalizar os dados para que cada pixel esteja entre 0 e 1
data = data.astype("float32") / 255.0

# Embaralhar os dados e labels
idxs = np.arange(0, len(data))
np.random.shuffle(idxs)
data = data[idxs]
labels = labels[idxs]

# Dividir os dados em conjuntos de treinamento e teste
split = int(0.8 * len(data))
train_data, test_data = data[:split], data[split:]
train_labels, test_labels = labels[:split], labels[split:]

# Definir a arquitetura da CNN
model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(img_size[0], img_size[1], 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

# Compilar o modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Treinar o modelo
history = model.fit(train_data, train_labels, validation_split=0.2, epochs=10)

# Fazer previsões no conjunto de teste
pred_labels = model.predict(test_data)
pred_labels = np.argmax(pred_labels, axis=1)

# Calcular a matriz de confusão e a acurácia
cm = confusion_matrix(test_labels, pred_labels)
acc = accuracy_score(test_labels, pred_labels)

# Salvar o modelo
model.save("classificador_imagens_2.h5")

# Imprimir a matriz de confusão e a acurácia
print("Matriz de Confusão:")
print(cm)
print("Acurácia: {:.2f}%".format(acc * 100))

