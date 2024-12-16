#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[4]:


import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, log_loss


# In[1]:


class FFNN:
    def __init__(self, layers, learning_rate=0.01, activation='relu', rho=0.0, lmbda=0.0, task_type='regression'):
        """
        Set up the neural network with the given parameters.
        
        Parameters:
        - layers: List of the number of neurons in each layer.
        - learning_rate: How fast the model learns; default is 0.01.
        - activation: Activation function for the hidden layers ('sigmoid', 'relu', 'leaky_relu').
        - rho: Momentum coefficient; if 0, no momentum.
        - lmbda: Regularization strength (L2 regularization to prevent overfitting).
        - task_type: 'regression' or 'binary_classification'.
        """
        self.layers = layers  # Store the structure of the network
        self.learning_rate = learning_rate  # Set the learning rate for weight updates
        self.rho = rho  # Momentum coefficient for weight updates
        self.lmbda = lmbda  # L2 regularization strength
        self.task_type = task_type  # Specify if we are doing regression or classification

        # Initialize lists to hold weights, biases, and other metrics
        self.weights = []
        self.biases = []
        self.velocities = []  # For momentum-based updates
        self.losses = []  # Track training losses
        self.val_losses = []  # Track validation losses
        self.train_accuracies = []  # Track training accuracy
        self.val_accuracies = []  # Track validation accuracy

        # Choose the right activation function and its derivative
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu
            self.activation_derivative = self.leaky_relu_derivative
        else:
            raise ValueError("Pick 'sigmoid', 'relu', or 'leaky_relu' as activation.")

        # Initialize weights and biases for each layer
        for i in range(len(self.layers) - 1):
            # Initialize weights with He initialization (good for ReLU)
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            bias_vector = np.zeros((1, self.layers[i + 1]))  # Biases start at zero
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
            self.velocities.append(np.zeros_like(weight_matrix))  # Initialize velocity to zero

    # Different activation functions and their derivatives
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid squashes values between 0 and 1

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # Derivative of the sigmoid function

    def relu(self, x):
        return np.maximum(0, x)  # ReLU keeps positive values, zeros out negatives

    def relu_derivative(self, x):
        return (x > 0).astype(float)  # ReLU derivative is 1 for positive, 0 otherwise

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)  # Leaky ReLU allows a small gradient when x < 0

    def leaky_relu_derivative(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha  # Derivative of Leaky ReLU uses alpha for negative x
        return dx

    # Forward pass: Go through each layer to calculate outputs
    def forward(self, x):
        activations = [x]  # Start with the input layer
        for i in range(len(self.weights) - 1):
            # Apply weights, biases, and activation function for each layer
            x = self.activation(np.dot(x, self.weights[i]) + self.biases[i])
            activations.append(x)  # Store activation for each layer

        # Final layer: Different output based on task type
        if self.task_type == 'binary_classification':
            x = self.sigmoid(np.dot(x, self.weights[-1]) + self.biases[-1])  # Sigmoid for binary classification
        else:
            x = np.dot(x, self.weights[-1]) + self.biases[-1]  # Linear output for regression
        activations.append(x)  # Add final output
        return activations  # Return all activations for backpropagation

    # Backward pass: Calculate gradients for weight updates
    def backward(self, y_true, activations):
        deltas = []  # Track "error signals" for each layer
        
        # Calculate error for the output layer
        if self.task_type == 'binary_classification':
            error = activations[-1] - y_true.reshape(-1, 1)  # Error for binary classification
            delta = error * self.sigmoid_derivative(activations[-1])  # Use sigmoid derivative for classification
        else:
            error = activations[-1] - y_true.reshape(-1, 1)  # Use MSE error directly for regression
            delta = error

        deltas.append(delta)  # Add output layer delta

        # Backpropagate through each hidden layer
        for i in reversed(range(len(self.weights) - 1)):
            error = np.dot(delta, self.weights[i + 1].T)  # Propagate error back to previous layer
            delta = error * self.activation_derivative(activations[i + 1])  # Apply activation derivative
            deltas.insert(0, delta)  # Insert at the beginning to keep the correct order

        return deltas  # Return all deltas for updating weights

    # Update weights and biases using calculated gradients
    def update_weights(self, activations, deltas, eta, epoch, batch_num, batch_size):
        for i in range(len(self.weights)):
            gradient = np.dot(activations[i].T, deltas[i]) + self.lmbda * self.weights[i]  # Gradient with regularization
            gradient = np.clip(gradient, -1, 1)  # Clip gradients to avoid exploding values

            # Apply momentum if specified
            if self.rho > 0:
                self.velocities[i] = self.rho * self.velocities[i] + eta * gradient  # Update velocity
                self.weights[i] -= self.velocities[i]  # Update weight using momentum
            else:
                self.weights[i] -= eta * gradient  # Standard weight update

            self.biases[i] -= eta * np.sum(deltas[i], axis=0, keepdims=True)  # Update bias

    # Learning rate scheduler
    def learning_schedule(self, t, t0=5, t1=10):
        return t0 / (t + t1)  # Decrease learning rate over time

    # Fit the model: Train on training data, optionally validate
    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=100, batch_size=10):
        """
        Train the model on the training data. Optionally validate with x_val and y_val.
        
        Parameters:
        - x_train: Training input data.
        - y_train: Training target data.
        - x_val: Validation input data (optional).
        - y_val: Validation target data (optional).
        - epochs: How many times to go through the data.
        - batch_size: Size of each training batch.
        """
        # Check input data for NaN or infinite values
        if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
            raise ValueError("Input data contains NaN values.")
        if np.any(np.isinf(x_train)) or np.any(np.isinf(y_train)):
            raise ValueError("Input data contains infinite values.")

        num_samples = x_train.shape[0]  # Get the number of training samples
        y_train = np.array(y_train)  # Ensure target data is a numpy array

        for epoch in range(epochs):
            # Shuffle the data each epoch for randomness
            indices = np.random.permutation(num_samples)  # Randomly shuffle indices
            x_shuffled = x_train[indices]  # Shuffle training data
            y_shuffled = y_train[indices]  # Shuffle target data

            for batch_num in range(0, num_samples, batch_size):
                # Get a batch of data
                x_batch = x_shuffled[batch_num:batch_num + batch_size]  # Current batch of input
                y_batch = y_shuffled[batch_num:batch_num + batch_size]  # Current batch of target values

                # Skip incomplete batches
                if len(y_batch) < batch_size:
                    continue

                # Perform forward and backward passes
                activations = self.forward(x_batch)  # Get activations from forward pass
                deltas = self.backward(y_batch, activations)  # Calculate deltas from backward pass
                self.update_weights(activations, deltas, self.learning_rate, epoch, batch_num, batch_size)  # Update weights

            # Calculate and track training accuracy and loss
            train_acc = self.accuracy(x_train, y_train)  # Compute training accuracy
            self.train_accuracies.append(train_acc)  # Store training accuracy for monitoring

            # Calculate and track loss
            final_activations = self.forward(x_train)[-1]  # Get the final activations for loss calculation
            epsilon = 1e-10  # Small constant to prevent log(0)

            # Loss calculation using log loss for binary classification
            if self.task_type == 'binary_classification':
                loss = log_loss(y_train, final_activations + epsilon)  # Use log loss for binary classification
            else:
                loss = np.mean((y_train - final_activations) ** 2)  # Mean squared error for regression

            # Add regularization to the loss
            loss += self.lmbda * np.sum([np.sum(w ** 2) for w in self.weights])  # Ridge regularization
            self.losses.append(loss)  # Store loss for tracking

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')

            # If we have validation data, track validation accuracy and loss
            if x_val is not None and y_val is not None:
                val_acc = self.accuracy(x_val, y_val)  # Calculate validation accuracy
                self.val_accuracies.append(val_acc)  # Store validation accuracy

                val_activations = self.forward(x_val)[-1]  # Get predictions for validation data
                if self.task_type == 'binary_classification':
                    val_loss = log_loss(y_val, val_activations + epsilon)  # Validation loss for classification
                else:
                    val_loss = np.mean((y_val - val_activations) ** 2)  # Validation loss for regression
                val_loss += self.lmbda * np.sum([np.sum(w ** 2) for w in self.weights])  # Regularization

                self.val_losses.append(val_loss)  # Store validation loss

                # Print validation metrics every 10 epochs
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}')

    # Predict the output for new input data
    def predict(self, x):
        """
        Make predictions based on input data.
        
        Parameters:
        - x: Input data to predict outputs for.

        Returns:
        - Predicted outputs.
        """
        predictions = self.forward(x)[-1]  # Get the output from the forward pass
        if self.task_type == 'binary_classification':
            # For binary classification, clip predictions to avoid issues and convert to binary
            predictions = np.clip(predictions, 1e-10, 1 - 1e-10)  # Prevent log(0) in binary case
            return (predictions >= 0.5).astype(int)  # Return class predictions based on threshold
        else:
            return predictions  # Directly return predictions for regression

    # Calculate accuracy for binary classification
    def accuracy(self, x, y):
        """
        Calculate accuracy of predictions.

        Parameters:
        - x: Input data to predict outputs for.
        - y: True labels to compare against.

        Returns:
        - Accuracy as a float.
        """
        y_pred = self.predict(x).reshape(-1)  # Flatten predictions to 1D
        y = np.array(y).reshape(-1)  # Flatten true labels to 1D

        # Check for NaN values in predictions or true labels
        if np.any(np.isnan(y_pred)) or np.any(np.isnan(y)):
            raise ValueError("Predictions or true labels contain NaN values.")

        # Ensure both predictions and labels are integers for comparison
        y_pred = y_pred.astype(int)
        y = y.astype(int)

        return np.mean(y_pred == y)  # Return the fraction of correct predictions

    # Evaluate the model's performance
    def evaluate(self, x, y):
        """
        Evaluate the model's performance on given data.
        
        Parameters:
        - x: Input data for evaluation.
        - y: True labels for evaluation.

        Returns:
        - Performance metrics based on task type.
        """
        y_pred_proba = self.predict(x)  # Get predicted probabilities
        y = np.array(y).reshape(-1)  # Flatten true labels

        if self.task_type == 'binary_classification':
            # For classification, calculate accuracy and binary cross-entropy loss
            accuracy = accuracy_score(y, (y_pred_proba >= 0.5).astype(int))  # Calculate accuracy from predicted classes
            bce = log_loss(y, y_pred_proba)  # Calculate binary cross-entropy loss
            print(f'Classification Accuracy: {accuracy:.4f}')
            print(f'Binary Cross-Entropy Loss: {bce:.4f}')
            return accuracy, bce
        else:
            # For regression, calculate mean squared error and R² score
            mse = mean_squared_error(y, y_pred_proba)  # MSE for regression
            r2 = r2_score(y, y_pred_proba)  # R² score
            print(f'Regression MSE: {mse:.4f}, R²: {r2:.4f}')
            return mse, r2



# In[ ]:




