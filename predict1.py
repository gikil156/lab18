import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load dữ liệu (Tự động gán nhãn từ tên thư mục)
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/dataset/train', 
    image_size=(150, 150), 
    batch_size=32
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/dataset/validation', 
    image_size=(150, 150), 
    batch_size=32
)

# 2. Xây dựng cấu trúc mạng CNN
model = models.Sequential([
    # Chuẩn hóa ảnh từ [0, 255] về [0, 1] để tính toán ổn định
    layers.Rescaling(1./255, input_shape=(150, 150, 3)), 
    # Tầng tích chập 1: Tìm đặc trưng cơ bản (cạnh, góc)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # Giảm kích thước ảnh
    # Tầng tích chập 2: Tìm đặc trưng phức tạp (hình khối)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Chuyển dữ liệu sang dạng vector để phân loại
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Lớp đầu ra (Sigmoid cho bài toán 2 lớp: chó/mèo)
    layers.Dense(1, activation='sigmoid')
])

# 3. Biên dịch mô hình (Thiết lập thuật toán học)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Huấn luyện (Cho máy học 5 vòng trên toàn bộ tập dữ liệu)
model.fit(train_ds, epochs=5, validation_data=val_ds)
