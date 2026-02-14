# Task 1: Predict if a person is healthy or not

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Create dataset: Features = [Exercise Hours, Water Intake], Label = Healthy (1) or Not (0)
X = np.array([[1, 2], [3, 1], [4, 4], [0, 1], [2, 3]])
y = np.array([[0], [1], [1], [0], [1]])

# 2. Split dataset into training (75%) and testing (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Build ANN
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))            # Output layer

# 4. Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train model
model.fit(X_train, y_train, epochs=100, verbose=1)

# 6. Predict on test data
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int)

# 7. Show results
print("Test Inputs:\n", X_test)
print("Predicted Probabilities:\n", predictions)
print("Predicted Class Labels:\n", predicted_classes)
