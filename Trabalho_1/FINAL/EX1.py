import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Carrega o conjunto de treinamento e teste
train_data = np.loadtxt('train_data.txt', delimiter=',')
test_data = np.loadtxt('test_data.txt', delimiter=',')

# Separa as entradas (características) dos rótulos (saídas)
X_train = train_data[:, :-2]
y_train = train_data[:, -1]

X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Define o modelo de rede neural
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(64, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(128, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(64, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(32, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compila o modelo
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Treina o modelo
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# Avalia o modelo usando o conjunto de teste
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Calcula e exibe a matriz de confusão e a taxa de acerto do modelo
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
print('Matriz de Confusão:')
print(cm)
print('Taxa de acerto: {:.2%}'.format(accuracy))

# Plota gráficos de evolução do modelo com as épocas
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Evolução da Perda')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Evolução da Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.show()

# Salvar o modelo
#model.save("modelo_EX1.h5")