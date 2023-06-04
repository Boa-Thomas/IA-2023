import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, SGD, Nadam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score

# Definir tamanho das imagens
img_size = (128, 128)

                #from google.colab import drive
                #drive.mount('/content/drive')

                # Definir diretório com as imagens
                #path = "/content/drive/MyDrive/UFSC/Semestre_8_2023_1/Inteligencia_Art/Trabalho_1/FINAL/IMAGES"

# Definir diretório com as imagens
path = "IMAGES"


# Definir lista com as classes
classes = ["OVELHA", "VACA"]

# Inicializar listas vazias para os dados e os labels
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
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
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

# Converter listas em arrays numpy
data = np.array(data)
labels = np.array(labels)

# Separar dataset em treinamento, validação e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=11)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=11)

# Definir modelo de rede neural convolucional

model = Sequential()
model.add(Conv2D(5, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(7, (3, 3),strides = 1,padding="same",activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(11, (3, 3),strides = 1,padding="same", activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid')) # Change this line


# Compilar modelo
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])


# Treinar modelo
history = model.fit(X_train, y_train, epochs=300, batch_size=16, validation_data=(X_val, y_val))

# Avaliar modelo no conjunto de teste
score = model.evaluate(X_test, y_test)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Change this line

# Calcular métricas de desempenho
cm = confusion_matrix(y_test, y_pred)  # Change this line
acc = accuracy_score(y_test, y_pred)   # Change this line

# Salvar modelo treinado
model.save("modelo_EX2_4096_2048_1024.h5")

# Imprimir métricas de desempenho
print("Acurácia: {:.2f}%".format(acc * 100))
print("Matriz de confusão:")
print(cm)

# Salvar modelo treinado
model.save("modelo_EX2_New.h5")

# Plot accuracy and loss graphs
import matplotlib.pyplot as plt

# Accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

