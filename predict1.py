# ==========================
# 1. IMPORT THƯ VIỆN
# ==========================

import tensorflow as tf              # Thư viện deep learning chính (Keras nằm trong TensorFlow)
import matplotlib.pyplot as plt     # Dùng để vẽ và hiển thị ảnh
from tensorflow.keras import datasets # Load dataset có sẵn (CIFAR-10)
import numpy as np                   # Xử lý mảng số (tensor, matrix)

# ==========================
# 2. LOAD MODEL ĐÃ TRAIN
# ==========================

# Load model CNN đã được huấn luyện trước và lưu lại
model = tf.keras.models.load_model("cnn_cifar10.keras")

# ==========================
# 3. LOAD DATA TEST
# ==========================

# CIFAR-10 gồm:
# - 50,000 ảnh train
# - 10,000 ảnh test
(_, _), (x_test, y_test) = datasets.cifar10.load_data()

# ==========================
# 4. CHUẨN HÓA ẢNH
# ==========================

# Đưa pixel từ [0..255] về [0..1] để model dễ học và dự đoán tốt hơn
x_test = x_test.astype("float32") / 255.0

# ==========================
# 5. DANH SÁCH CLASS
# ==========================

# CIFAR-10 có 10 loại đối tượng
class_names = [
    "airplane",   # 0
    "automobile", # 1
    "bird",       # 2
    "cat",        # 3
    "deer",       # 4
    "dog",        # 5
    "frog",       # 6
    "horse",      # 7
    "ship",       # 8
    "truck"       # 9
]

# ==========================
# 6. CHỌN 1 ẢNH ĐỂ TEST
# ==========================

index = 0  # chọn ảnh thứ 0 trong tập test

image = x_test[index]          # lấy ảnh input
actual_label = y_test[index][0] # lấy nhãn thật của ảnh

# ==========================
# 7. CHUẨN BỊ INPUT CHO MODEL
# ==========================

# Model cần input dạng batch (nhiều ảnh cùng lúc)
# nên ta thêm 1 chiều batch
image_batch = np.expand_dims(image, axis=0)

# ==========================
# 8. DỰ ĐOÁN
# ==========================

# model trả về xác suất cho 10 class
prediction = model.predict(image_batch)

# lấy index có xác suất cao nhất
predicted_label = np.argmax(prediction)

# ==========================
# 9. IN KẾT QUẢ
# ==========================

print("Actual Label:")
print(class_names[actual_label])  # nhãn thật

print("\nPredicted Label:")
print(class_names[predicted_label])  # nhãn model đoán

print("\nProbabilities:")
print(prediction)  # xác suất từng class

# ==========================
# 10. HIỂN THỊ ẢNH
# ==========================

plt.imshow(image)  # hiển thị ảnh

# đặt tiêu đề gồm:
# - nhãn thật
# - nhãn dự đoán
plt.title(
    f"Actual: {class_names[actual_label]}\n"
    f"Predicted: {class_names[predicted_label]}"
)

plt.axis("off")  # tắt trục tọa độ cho đẹp
plt.show()       # hiển thị hình
