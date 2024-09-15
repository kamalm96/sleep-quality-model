# Sleep Quality Prediction Model
## This project demonstrates a machine learning model to predict sleep quality based on user health and lifestyle data.

## Overview
- Data Preparation: Cleaned and preprocessed data by converting time features (e.g., bedtime, wake-up time) to minutes past midnight, encoding categorical variables (e.g., gender, activity level), and normalizing numerical features (e.g., age, steps).
- Model Architecture: Built a neural network using Keras with two hidden layers (64 and 32 neurons) and ReLU activation. The output layer uses a linear activation function for regression.
- Training: Compiled the model with the Adam optimizer and Mean Squared Error (MSE) loss function. Trained over 100 epochs with batch size 32, using validation data to monitor performance.
- Results: Achieved a Mean Absolute Error (MAE) of 0.114 on the test set, indicating good predictive accuracy.
- Visualization: Plotted training and validation loss to show model convergence without overfitting.
## How to Use
- Clone the repository.
- Install dependencies: pip install -r requirements.txt.
- Run train_model.py to train the model.
- Use sleep-quality.py to make predictions with new data. (python sleep-quality.py)
![image](https://github.com/user-attachments/assets/541715b3-fac9-44d9-a86f-71a607babf18)
