# Sleep Quality Prediction Model
## This project demonstrates a machine learning model to predict sleep quality based on user health and lifestyle data.
### _Will be adding a way to use this model soon within a React project_

## Overview
- Data Preparation: Cleaned and preprocessed data by converting time features (e.g., bedtime, wake-up time) to minutes past midnight, encoding categorical variables (e.g., gender, activity level), and normalizing numerical features (e.g., age, steps).
- Model Architecture: Built a neural network using Keras with two hidden layers (64 and 32 neurons) and ReLU activation. The output layer uses a linear activation function for regression.
- Training: Compiled the model with the Adam optimizer and Mean Squared Error (MSE) loss function. Trained over 100 epochs with batch size 32, using validation data to monitor performance.
- Results: Achieved a Mean Absolute Error (MAE) of 0.114 on the test set, indicating good predictive accuracy.
- Visualization: Plotted training and validation loss to show model convergence without overfitting.
## How to Use
- Clone the repository.
- Install dependencies: pip install -r requirements.txt.
- Run sleep-quality.py to train the model. (python sleep-quality.py)
## Training curve
![image](https://github.com/user-attachments/assets/541715b3-fac9-44d9-a86f-71a607babf18)
