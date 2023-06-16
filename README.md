# Guard+ Machine Learning

## Abstract

The Guard+ project incorporates machine learning techniques to enhance the mobile application's capabilities in providing crucial information regarding potential earthquakes and volcanic eruptions. Our approach involved a comprehensive ML plan and workflow to develop a classification model that enables efficient risk assessment and preparedness for individuals visiting unfamiliar locations. 

By leveraging TensorFlow and Keras frameworks, we designed a binary classification neural network capable of making accurate predictions. The trained model is integrated into the Guard+ app, empowering users with real-time insights and practical guidance to mitigate the impact of natural disasters. This abstract highlights our ML work and the innovative ideas driving the development of Guard+ to ensure the safety and well-being of individuals exploring new environments.

## Machine Learning

The machine learning component of Guard+ focuses on developing a classification model to enhance the application's capabilities. The following steps were undertaken to accomplish this:

### Data Processing

- The dataset was preprocessed, including one-hot encoding, to prepare it for model training and evaluation.
- Categorical columns were one-hot encoded using the onehot_encoder function to convert them into binary vectors.
- Numerical columns were standardized using the StandardScaler from scikit-learn to improve model performance.

### Data Split

- The preprocessed data was split into training and testing sets using the train_test_split function from scikit-learn.
- Input features (X) were assigned all columns except the target variable ('Status'), and the target variable (y) was assigned the 'Status' column.

### Model Architecture

- A neural network model was created using the Sequential class from Keras.
- The model consisted of an input layer with a shape corresponding to the number of input features.
- Two dense layers with ReLU activation functions were added after the input layer to introduce non-linearity and capture complex relationships in the data.
- The final layer had a single neuron with a sigmoid activation function for binary classification.

### Model Compilation and Training

- The model was compiled with an optimizer (Adam), a loss function (binary cross-entropy), and a metric (AUC - Area Under the Curve).
- Training was performed using the fit method, providing the training data (X_train and y_train).
- A validation split of 0.2 was used to allocate 20% of the training data for validation during the training process.
- The batch size determined the number of samples processed before updating the model's internal parameters.
- The number of epochs defined the number of times the model iterated over the entire training dataset.
- Callbacks were set up to monitor validation AUC, apply early stopping, save the best weights, and reduce the learning rate if necessary.

### Model Evaluation

- After training, the model was evaluated using the test data (X_test and y_test) with the evaluate method.
- The evaluation provided the loss value and the AUC metric as performance metrics.

### Predictions and Confusion Matrix

- The model was used to make predictions on the test data using the predict method.
- A threshold of 0.5 was applied to convert the predictions into binary values (0 or 1).
- A confusion matrix was created using the true labels (y_test) and the predicted labels.
- The confusion matrix was visualized using a heatmap generated with Seaborn.

### Model Saving

- The trained model was saved using the pickle module in Python and dumped into a file called 'model'.
- This allows the model to be reused and integrated into the Guard+ app.

This README provides an overview of the machine learning work conducted for the Guard+ project. It outlines the steps involved in data processing, model architecture, training, evaluation, and model saving. The integration of machine learning capabilities in Guard+ aims to provide users with real-time insights and actionable information to ensure their safety and preparedness when visiting areas prone to earthquakes and volcanic eruptions.
