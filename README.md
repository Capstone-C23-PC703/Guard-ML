# Guard+ Machine Learning

Guard+ is a mobile application developed to provide information about potential risks of earthquakes and volcanic eruptions, primarily targeting tourists and individuals visiting new places. The application aims to minimize the impact of such disasters by enabling users to be better prepared and informed before visiting areas prone to such events. To enhance the application's functionality, a machine learning model has been developed to provide accurate predictions.

## Machine Learning Model

The machine learning model in Guard+ is designed to classify earthquake and volcanic eruption risks. Here's an overview of how the model works:

### Data Processing

The dataset used for training and testing the model is processed before training. This involves various steps such as data cleaning, feature engineering, and data normalization.

- The dataset was preprocessed, including one-hot encoding, to prepare it for model training and evaluation.
- The preprocessed data was split into training and testing sets using the train_test_split function from scikit-learn.

### Neural Network Architecture

The model architecture is based on a neural network created using TensorFlow and Keras. The neural network consists of multiple layers, including input, hidden, and output layers. The number of neurons in each layer can be adjusted based on the desired complexity of the model.

- Two dense layers with ReLU activation functions were added after the input layer to introduce non-linearity and capture complex relationships in the data.
- The final layer had a single neuron with a sigmoid activation function for binary classification.

### Training the Model

The model is trained using labeled data, where features are the inputs related to earthquake and volcanic eruption characteristics, and the target variable is the corresponding risk classification. The training process involves optimizing the model's parameters using an optimization algorithm and minimizing a defined loss function.

- The model was compiled with an optimizer (Adam), a loss function (binary cross-entropy), and a metric (AUC - Area Under the Curve).
- A validation split of 0.2 was used to allocate 20% of the training data for validation during the training process.

### Evaluation and Performance Metrics

After training, the model's performance is evaluated using a separate test dataset. Performance metrics such as accuracy, precision, recall, and F1-score are computed to assess the model's effectiveness in predicting earthquake and volcanic eruption risks.

- The model was used to make predictions on the test data using the predict method.
- The confusion matrix was visualized using a heatmap generated with Seaborn.

### Saving the Model

Once the model is trained and evaluated, it is saved for future use in the Guard+ application. The trained model is serialized using the pickle module in Python and stored in a file for easy retrieval and deployment.

## Integration with Guard+

The trained machine learning model is integrated into the Guard+ application, allowing users to access real-time risk predictions based on location and other relevant factors. The model provides valuable insights to help users make informed decisions and take necessary precautions to mitigate the impact of earthquakes and volcanic eruptions.
