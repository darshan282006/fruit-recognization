# fruit-recognization

!pip install tensorflow


import tensorflow as tf
print(tf.__version__)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("fruit.csv")

# strip any weird spaces in column names/values
data.columns = data.columns.str.strip()
data['fruit_name'] = data['fruit_name'].str.strip()
data['fruit_subtype'] = data['fruit_subtype'].str.strip()

print(data.head())
print(data.info())

# Features & Labels
X = data[['mass', 'width', 'height', 'color_score']].values   # only numeric features
y = data['fruit_name'].values                                 # classify based on fruit name

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Normalize inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# Save model
model.save("fruit_classifier_tabular_fixed.h5")

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()
