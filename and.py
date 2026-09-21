import numpy as np

from main import NeuralNetwork  

if __name__ == "__main__":
    # Input data for AND gate
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [0],
                  [0],
                  [1]])
    
    print("Initializing Neural Network for AND gate...")

    #Instantiate the network
    network_and = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    #Training
    network_and.training(X=X, y=y, epochs=10000, learning_rate=1.0)

    #save the model
    network_and.save_model("and_model")

    print("Final results...")
    for i in range(len(X)):
        prediction = network_and.feedforward(X[i:i+1])
        predicted_value = prediction[0][0]
        expected_value = y[i][0]
        decision = 1 if predicted_value >= 0.5 else 0

        print(f"Input: {X[i]} - Predicted: {predicted_value:.4f} - Expected: {expected_value} - Decision: {decision}")