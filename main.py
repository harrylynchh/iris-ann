'''
main.py
5/9/2025
Harry Lynch
Main entry point for the Iris flower classifier which preprocesses the dataset, 
allows the user to select the learning rate and epoch count (or use a tuned default) 
before training the model.  After training and displaying training data, prompt the
user loopily for their own entries to be ran through the model giving a classification.  
'''
import numpy as np
from preprocessor import load_and_preprocess
from neural_network import NeuralNetwork
from globals import IDX_TO_LABELS

# Number of neurons foreach layer in the network (input->hidden->output)
# NOTE: THIS WAS DETERMINED THRU TRIAL AND ERROR (for the hidden layer that is)
NETWK_DIMS = [4, 8 ,3]

# Preprocess data and separate into training, validation, and testing sets
# NOTE: We retrieve the scaler object s.t we can identically fit given data points
#       from user
X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = load_and_preprocess('iris_dataset.txt')

# Testing var so I could skip user input and just see the model run on tuned settings
testing = False
if not testing:
    print("Welcome to the Iris Assessor- this model usually operates, depending on learning rate and epoch count, \nat around 93-96% (and sometimes even to 100%) accuracy on the Testing Set.  Select model design choices or just hit enter to use the tuned defaults\n")

    l_rt = input("What would you like the learning rate to be? NOTE: THERE IS 50% DECAY EVERY 25 EPOCHS (Default 0.1): ")
    l_rt = 0.1 if l_rt == "" else float(l_rt)

    ep_ct = input("How many epochs would you like to train with? (Default 100): ")
    ep_ct = 100 if ep_ct == "" else int(ep_ct)
else:
    ep_ct = 100
    l_rt = 0.1

# Initialize network object to train
nn = NeuralNetwork(NETWK_DIMS, lr=l_rt)
print("===================== BEGIN TRAINING ======================")
# Train the model and then test against test-set, reporting accuracy as %
nn.train(X_train, Y_train, X_val, Y_val, epochs=ep_ct)
preds = nn.predict(X_test)
# Calculate by creating tuples of (prediction_idx, label_idx) foreach prediction and comparing
acc = sum(int(p==t) for p,t in zip(preds, np.argmax(Y_test, axis=1))) / len(preds)
print(f'Test Accuracy: {acc:.4f}')
print("====================== END TRAINING =======================")

# Give user option to enter their own datapoints
if not testing:
    print("Now that the model is trained, feel free to input the measurements of your own mystery Iris in the following format:")
    print("{SEPAL LENGTH (cm)},{SEPAL WIDTH (cm)},{PETAL LENGTH (cm)},{PETAL WIDTH (cm)} (e.g '5.1,3.5,1.4,0.2')")
    user_feats: str = ""
    while user_feats != 'q': 
        user_feats = input("Your plant ('q' to quit): ").split(',')
        
        if 'q' in user_feats:
            break
        
        # Convert to floats
        user_feats = [float(x) for x in user_feats]
        
        if len(user_feats) != 4:
            print(f"ERROR: Expected 4 values, got {len(user_feats)} values of {user_feats}.")
            continue
        
        # Scale the input the same way we did the original dataset for z-score (see preprocessor fn)
        user_scaled = scaler.transform([user_feats])
        # Run thru the predictor and print the result
        pred = nn.predict(user_scaled)[0]
        print(f"The model thinks your flower is a: {IDX_TO_LABELS[pred]}\n")
    print("Thanks for classifying- hope we got it right.")