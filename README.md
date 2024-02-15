# sentiment-analysis-using-LSTM
## Description-
The provided code preprocesses IMDb movie review data, tokenizes and pads the text sequences, builds a bidirectional LSTM model to classify sentiment (positive or negative), and evaluates the model's performance on a test set. The preprocessing includes removing HTML tags, URLs, stopwords, and lemmatizing text. After model training, it predicts sentiments for new sentences and prints the results. Overall, the code demonstrates text preprocessing, model building, training, and evaluation for sentiment analysis on IMDb reviews.
## Explanation-
  -Imports: The code imports necessary libraries such as re for regular expressions, pandas for data manipulation, numpy for numerical operations, and various modules from sklearn and keras for machine learning tasks.

  -Data Preprocessing: The code reads a CSV file containing IMDb movie reviews and their sentiments. It preprocesses the reviews by removing HTML tags, URLs, stopwords, and non-alphanumeric characters. It also lemmatizes the text to reduce words to their base form.

  -Data Statistics: The code calculates the average length of each review and the percentage of positive and negative sentiment reviews in the dataset.

  -Label Encoding: It encodes the sentiment labels ('positive' and 'negative') into numerical values using LabelEncoder from sklearn.

  -Train-Test Split: It splits the dataset into training and testing sets.

  -Model Configuration: Hyperparameters such as vocabulary size, embedding dimension, and maximum sequence length are defined. A tokenizer is created to convert text data into sequences and pad them to a fixed length.

  -Model Definition: A sequential Keras model is constructed with an embedding layer, bidirectional LSTM layer, and dense layers with activation functions.

  -Model Compilation: The model is compiled with binary cross-entropy loss and Adam optimizer.

  -Model Training: The model is trained on the training data for a specified number of epochs with validation split.

  -Prediction and Evaluation: The trained model is used to predict sentiments on the test set. Accuracy is calculated using sklearn's accuracy_score function. Additionally, the model is applied to new sentences to predict their sentiment.
