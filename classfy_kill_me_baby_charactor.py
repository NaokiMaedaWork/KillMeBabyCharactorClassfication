import os
import tensorflow as tf
import numpy as np
import cv2

def showImage(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

#画像読み込み
train_img_dirs = ['yasuna', 'sonya', 'agiri','other']

train_image = []
train_label = []

for i, folder in enumerate(train_img_dirs):
    files = os.listdir('./' + folder)
    for image in files:
        # 画像読み込み
        img = cv2.imread('./' + folder + '/' + image)
        # 28x28にリサイズ
        img = cv2.resize(img,(28,28))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train_image.append(img)

        # ラベルを入れる
        train_label.append(i)

#numpy配列に変換
train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
          
#モデル定義
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_image,train_label,epochs=5)

# テスト
files = os.listdir('./testimage')
for image in files:
  img = cv2.imread('./testimage/' + image)
  showImage(img)
  img = cv2.resize(img,(28,28))
  img = (np.expand_dims(img,0))
  print('映っているのは' + train_img_dirs[model.predict(img)[0].argmax()])

