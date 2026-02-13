# Task 2: Predict if a student will pass or fail

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Dataset: Features = [Assignments Submitted, Attendance %], Label = Pass(1)/Fail(0)
X = np.array([[4, 80], [2, 60], [5, 90], [1, 50], [3, 70]])
y = np.array([[1], [0], [1], [0], [1]])

# 2. Split dataset into training and testing (75% train, 25% test)
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
