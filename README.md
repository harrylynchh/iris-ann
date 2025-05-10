# ANN Iris Flower Classifier

## Harry Lynch

### 5/9/2025

---

# About:

    This program is designed to classify an Iris flower by Sepal length/width and
    Petal length/width (both in cm) to one of three species:
        - Iris Setosa
        - Iris Versicolour
        - Iris Virginica
    I accomplished this by utilizing a fully-connected feedforward Artificial
    Neural Network (ANN) which attempts to generalize the trends in the above features
    to classify the flower.

    Given 150 Samples (50/class), I used a 60/20/20 split (training/validation/testing)
    with each of those sets maintaining balance and proper stratification of the class.
    I used z-score normalization using scikit's StandardScaler to ensure that the data
    had a mean of 0 and stdev of 1 before training on it.  I applied this same technique
    to any of the user-inputted statistics such that it fit the rest of the dataset
    (more on StandardScaler below).  I then used scikit's train_test_split function
    to execute the aforementioned stratified partitioning of the data, this made it
    easy to just pass the proporiton of the data I wanted into the set and it also
    separated out the labels and features into two separate numpy arrays.  This
    concluded the preprocessing portion of the project.

    After preprocessing, the program gives the user the opportunity to select a custom
    learning rate and epoch count (see Usage), and then begins building the Network.
    I landed on a 4-8-3 setup after some trial/error for determining the number
    of hidden layer neurons.  For the learning rate, I initially had a very low
    static learning rate but decided to implement decay as it was relatively
    simple to add on and improved my accuracy.  I settled on a 50% reduction every 25
    epochs (100 total epochs) which was very simple and did what I needed it to do with a starting lr of
    0.1.  Training-wise, everything is pretty standard with the addition of shuffling
    the order of the training set with each epoch.  I saw a bump in performance as a
    result of this and decided to leave it in.  After each iteration of forward/back
    propagation, I predict against the validation set and report accuracy per epoch.
    This gave me a good idea of when I was bottoming out and while I experimented with
    early-stop (see Resources).

    Structurally, the program is broken down into two classes, NeuralNetwork and
    Layer with the former being comprised of an array of the latter.  Both have
    named implementations of fwd/backward propagation with the network serving as
    a facilitator passing the activations (fwd) and gradients (backward) between
    the layers.  The Layers themselves handle the gradient calculations, activations
    and weight/bias adjustments.  Each neuron has it's own bias and weight associated
    with it represented in two arrays W and b within each layer.  Root Mean Square
    Error was the loss function I chose as it was what we were introduced to although
    I've read that there might be more room for improvement (see last paragraph).

    After training, I run against the test set and report accuracy one final time
    before prompting the user for comma separated features for their own samples.
    This step simply takes their four given features, applies the same scaling
    using the same StandardScaler from the preprocessor (z-score normalization)
    and run an instance of forward propogation with the features as inputs using
    the NN's predict function.  I then relay the classification back to the user
    before loopily prompting again until quit (once again, see Usage).

    Overall the Network performs pretty well against randomly generated training/validation/testing
    sets with accuracy between 90-100% with most sets landing testing accuracy
    around 93-96%.  I believe there are some improvements to be made as I read a
    bit about cross-entropy and softmax as a combination of output activation and
    loss function (and I believe it was briefly mentioned in class).  As I didn't quite
    understand it I chose not to implement it here but I did find it worth mentioning.
    To the same end, my decay schedule is primitive but for how simple the problem
    space is I think it accomplishes the job over fancier equations.  I'm happy with
    how it performs and I learned a lot in the process of it's creation.

---

# Usage:

    - This program relies on only two dependencies for data processing
      and representation (numpy and scikit) before running, to run,
      start up a virtual environment of your choosing and run:
              ** pip install < requirements.txt **
      This will install all dependencies listed in requirements.txt

    - After the environment is set up, to train and prompt the model run the following
                                ** python main.py **
      NOTE: Training took a very short amount of time for me (a few seconds)
      but this could be due to the quality of my PC so apologies if it takes awhile
      to train.

    - The program will process, train, and take sample prompts from the user outputting
      the following:
        - Epoch progress counter and Validation accuracy after each epoch.
        - After training, final test-set accuracy.
        - Foreach sample the user provides formatted comma separated as:
         ** {SEPAL LENGTH},{SEPAL WIDTH},{PETAL LENGTH},{PETAL WIDTH} **
          with all units in centimeters.  The model will predict it's identity
          and relay the name of the species back to the user.
        - Cease program by pressing 'q' instead of entering a sample.

# Resources:

    - I referenced StackOverflow a few times for some help with SciKit aswell as the official
      SciKit api linked below.
    - I extensively referenced the numpy documentation throughout this project as
      most of the manipulation is done through numpy arrays.
    - I used two major implements from SciKit, StandardScaler and train_test_split, the
      references for both I used heavily (all linked below)
    - Both ANN Presentations (Especially for the backprop implementation) were helpful
      for designing and implementing the

See linked here:
[NumPy](https://numpy.org/doc/stable/reference/arrays.ndarray.html),
[SciKit](https://scikit-learn.org/stable/api/index.html),
[Early Stopping](https://en.wikipedia.org/wiki/Early_stopping),
[train_test_split SciKit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
[StandardScaler SciKit](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
External packages required to run this program are listed in `requirements.txt`

This is what I referred to earlier I thought it was interesting so I'll include:
[Softmax Cross-Entropy Combo (way too complex for me to use here)](https://www.parasdahal.com/softmax-crossentropy).
