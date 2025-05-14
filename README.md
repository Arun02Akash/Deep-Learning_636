To generate predictions, open the Deep_Learning_Final_Project_3.ipynb notebook located in the DL_Project_Final_3 directory using Google Colab. Then, mount the entire zip file. Once mounted, run the last two cells of the Testing section to perform inference.

Each folder within the project corresponds to a specific (n, k) configuration and contains the associated classifier model, encoder-decoder model, scaler, and regressor models needed for prediction.

📚 Model Architecture Overview

We build a three-stage system for each (n, k) setup:

1. Encoder-Decoder:
  - Purpose: Compress the full input vector [n, k, m, P_augmented] into a lower-dimensional latent representation.
  - Encoder: Fully connected layers with ReLU activations, ending in a latent space.
  - Decoder: Expands from latent space back to an output vector during training (not used in final inference).
  - After training, we retain only the encoder for feature extraction.

2. Classifier (MLP Classifier):
  - Purpose: Classify the latent compressed representation into one of 20 quantile-based classes.
  - Structure:
  - 1 hidden layer
  - ReLU activation
  - Output layer with 20 neurons (for 20 class labels).
  - The classifier decides which regressor should be used for final prediction.

3. Regressors (One MLP Regressor per Class):
  - Purpose: Predict the final continuous result value based on the compressed representation.
  - Structure:
   - Separate MLP regressor for each class.
   - 1 or more hidden layers (varying depth depending on (n,k)).
   - Output is a single neuron predicting log2(result).
  - After prediction, we invert the log2 to obtain the actual result (result = 2^(predicted log2)).

🛠 Full Inference Pipeline:
  - Input sample -> Add (n, k, m) -> Flatten -> Scale.
  - Pass through the trained Encoder to get compressed representation.
  - Use Classifier to predict the class.
  - Pass compressed representation through the corresponding class-specific Regressor.
  - Invert the log2 transformation to recover the final predicted result.

🚀 This modular design ensures better specialization for different regions of the target space, leading to more accurate and robust predictions.
