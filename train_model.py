import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary", shuffle=False
)

class_names = train_ds.class_names
print("Classes (0/1 mapping):", class_names)

def preprocess(x, y):
    return preprocess_input(x), y

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)
test_ds_raw = test_ds
test_ds = test_ds.map(preprocess)

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_true = np.concatenate([y.numpy() for x, y in test_ds_raw], axis=0).flatten()
y_pred_prob = model.predict(test_ds)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend(); plt.title("Loss")
plt.savefig("models/training_history.png")
print("Saved training curves to models/training_history.png")

model.save("models/parkvision_model.h5")
print("Model saved to models/parkvision_model.h5")
