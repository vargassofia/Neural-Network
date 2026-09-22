import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        #np.random.seed(42) #So that the random results are always the same
        #Hidden Layer
        #Weight matrix W1 of size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        #Size b1 bias vector
        self.b1 = np.zeros((1, hidden_size))

        #Output layer
        #Weight matrix W2 
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        # size b2 bias vector
        self.b2 = np.zeros((1, output_size))

    def _sigmoide(self, x):
        #Non-linear activation function
        return 1 / (1 + np.exp(-x))
    
    def _derivada_sigmoide(self, a):
        #derivative of the sigmoid a * (1 - a)
        return a * (1 - a)
    def feedforward(self, X):
        #Input hidden layer 
        #Z1 = X * W1 + b1
        self.z1 = np.dot(X, self.W1) + self.b1
        #The activation function is applied
        self.a1 = self._sigmoide(self.z1)

        #Output hidden layer
        #Z2 = A1 * W2 + b2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        #The activation function is applied to obtain the prediction
        self.a2 = self._sigmoide(self.z2)

        return self.a2
        
    def backpropagation(self, X, y, learning_rate):
        # calculate the error in the output layer
        error_output = y - self.a2
        # calculate the Delta of the output
        delta_output = error_output * self._derivada_sigmoide(self.a2)
        # calculate the error in the hidden layer
        error_hidden = np.dot(delta_output, self.W2.T)
        # Calculate the Delta of the hidden layer
        delta_hidden = error_hidden * self._derivada_sigmoide(self.a1)
        
        # We adjusted W2 and b2 (Output layer weights)
        self.W2 += np.dot(self.a1.T, delta_output) * learning_rate
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * learning_rate

        # We adjusted W1 and b1 (Hidden layer weights)
        self.W1 += np.dot(X.T, delta_hidden) * learning_rate
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate
    def training(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)

            #MSE: Mean squared error, diference between the expected value and the predicted value
            if epoch % 1000 == 0:
                error = np.mean(np.square(y - self.a2))
                print(f"Epoch{epoch} - Error: {error:.6f}")

    def save_model(self, file_name):
        # np.savez saves the model parameters to a file
        np.savez(file_name, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Model saved to", file_name)  

    def load_model(self, file_name):
        # np.load loads the model parameters from a file
        data = np.load(f"{file_name}.npz")
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Model loaded successfully from {file_name}.npz")
