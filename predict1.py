import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import numpy as np

# 1. Tải và chuẩn hóa dữ liệu
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # Chuẩn hóa về [0, 1]
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 2. Xây dựng cấu trúc mạng CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax') # Đầu ra 10 lớp
])

# 3. Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Huấn luyện
history = model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2, verbose=2)

# 5. Đánh giá và Vẽ đồ thị
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Độ chính xác trên tập test: {test_acc:.3f}")

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy qua các Epoch')
plt.show()
