import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. 데이터 로드 (CIFAR-10)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 0~255 → 0~1 범위로 데이터 정규화
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# CIFAR-10 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. CNN 모델 구성
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 모델 학습
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

# 5. 성능 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"테스트 정확도: {test_acc:.4f}")

test_batch = test_images[:10]
predict = model.predict(test_batch)
print("predict : ", predict)

# 6. 학습 곡선 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#### 예측######
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# CIFAR-10 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 1. 이미지 로드 & 리사이즈
img_path = "img.png"  # 여기에 개인 이미지 경로
img = image.load_img(img_path, target_size=(32, 32))  # CIFAR-10 크기

# 2. 배열로 변환 & 정규화
img_array = image.img_to_array(img) / 255.0  # 0~1 스케일
img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

# 3. 예측
predictions = model.predict(img_array)
predicted_label = np.argmax(predictions)

# 4. 결과 출력
plt.imshow(image.load_img(img_path))
plt.title(f"Predicted: {class_names[predicted_label]}")
plt.axis('off')
plt.show()