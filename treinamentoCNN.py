import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

### Parâmetros
path = "Imagens"  # Diretório com suas imagens
batch_size_val = 50
epochs_val = 10
imageDimesions = (32, 32, 3)  # Dimensões das imagens

## Importar Imagens
count = 0
images = []
classNo = []
pastas = os.listdir(path)
print("Total de Classes:", len(pastas))
noOfClasses = len(pastas)

for pt in range(0, len(pastas)):
    arquivos = os.listdir(path + "/" + str(count))
    for arq in arquivos:
        curImg = cv2.imread(path + "/" + str(count) + "/" + arq)
        images.append(curImg)
        classNo.append(count)
    count += 1

images = np.array(images)
classNo = np.array(classNo)

## Separando Imagens em treino, validação e teste
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

## Funções do pré-processamento das Imagens
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)     # Converter em Gray
    img = equalize(img)      # Padronizar a Luminosidade
    img = img / 255          # Normalizar valores entre 0 e 1
    return img

## Pré-processar imagens
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

## Regularizar Arrays (reshape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

## Aumentar imagens com ImageDataGenerator
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

X_batch, y_batch = next(dataGen.flow(X_train, y_train, batch_size=20))

## One-hot encoding
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

## Calcular `steps_per_epoch`
steps_per_epoch_val = len(X_train) // batch_size_val  # Ajuste para garantir que não passe dos dados

## Criar o Modelo
def myModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

## Treinamento do Modelo
model = myModel()
print(model.summary())

history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,  # Isso agora será 14, conforme o tamanho dos dados
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=True
)

## Mostrar histórico de treinamento
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

## Avaliar o Modelo
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

## Salvar o modelo
model.save('modelo.keras')
print('Modelo Salvo!')
