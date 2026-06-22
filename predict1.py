import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
import numpy as np

# ==========================
# LOAD TRAINED MODEL
# ==========================

model = tf.keras.models.load_model(
    "cnn_cifar10.keras"
)

# ==========================
# LOAD TEST DATA
# ==========================

(_, _), (x_test, y_test) = datasets.cifar10.load_data()

x_test = x_test.astype("float32") / 255.0

# ==========================
# CLASS NAMES
# ==========================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# ==========================
# SELECT IMAGE
# ==========================

index = 0

image = x_test[index]

actual_label = y_test[index][0]

# ==========================
# PREDICT
# ==========================

image_batch = np.expand_dims(
    image,
    axis=0
)

prediction = model.predict(
    image_batch
)

predicted_label = np.argmax(
    prediction
)

# ==========================
# PRINT RESULT
# ==========================

print("Actual Label:")
print(class_names[actual_label])

print("\nPredicted Label:")
print(class_names[predicted_label])

print("\nProbabilities:")
print(prediction)

# ==========================
# DISPLAY IMAGE
# ==========================

plt.imshow(image)

plt.title(
    f"Actual: {class_names[actual_label]}\n"
    f"Predicted: {class_names[predicted_label]}"
)

plt.axis("off")
plt.show()