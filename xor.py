import numpy as np
from main import NeuralNetwork

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]]) 

    # PHASE 1: TRAIN AND SAVE

    print("PHASE 1: Training the original network")
    original_network = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    original_network.training(X=X, y=y, epochs=10000, learning_rate=1.0)
    
    # Save the learned weights
    original_network.save_model("xor_model")

    # PHASE 2: LOAD AND PREDICT (Without training)

    print("PHASE 2: Testing the memory of a new network")
    # Instantiate a completely new network (with useless random weights)
    cloned_network = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # Inject the saved knowledge
    cloned_network.load_model("xor_model")
    
    print("Results of the cloned network (without having been trained):")
    for i in range(len(X)):
        prediction = cloned_network.feedforward(X[i:i+1])
        predicted_value = prediction[0][0]
        expected_value = y[i][0]
        decision = 1 if predicted_value >= 0.5 else 0 
        print(f"Input: {X[i]} - Predicted: {predicted_value:.4f} - Expected: {expected_value} - Decision: {decision}")