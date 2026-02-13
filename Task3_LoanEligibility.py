# Task 3: Predict if a person is eligible for a loan

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Dataset: Features = [Monthly Income, Credit Score], Label = Eligible(1)/Not Eligible(0)
X = np.array([[5000, 700], [3000, 600], [7000, 750], [2000, 550], [4000, 680]])
y = np.array([[1], [0], [1], [0], [1]])

# 2. Split dataset into training and testing (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Build ANN
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # Hidden layer with 4 neurons
model.add(Dense(1, activation='sigmoid'))            # Output layer for binary classification

# 4. Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# 6. Predict on unseen test data
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int)

# 7. Display results
print("Test Inputs:\n", X_test)
print("Predicted Probabilities:\n", predictions)
print("Predicted Class Labels (0 or 1):\n", predicted_classes)
