# Overview
Sentiment analysis is a natural language processing (NLP) task focused on identifying and categorizing sentiments expressed in text. In this project, I used a dataset from Kaggle containing text data labeled with sentiment classes to build a deep learning model for automatic sentiment classification.

# Steps Involved

## 1. Data Preprocessing
Before feeding text into a deep learning model, it's crucial to clean and preprocess it to ensure high-quality input. Here’s the preprocessing pipeline I followed:

Lowercasing: Converted all text to lowercase to ensure uniformity, treating "Happy" and "happy" as the same word.
Removing Special Characters and Numbers: Removed punctuation marks and numbers to reduce noise and focus on meaningful words.
Tokenization: Split text into individual words or tokens for easier handling by the model.
Stop Words Removal: Eliminated common but unimportant words like "the," "is," and "on" using a predefined set of stop words.
Stemming: Reduced words to their base or root form (e.g., "running" to "run") to group different forms of a word together and decrease vocabulary size.

## 2. Splitting the Dataset
Post-preprocessing, the dataset was divided into training and test sets:

Training Data: 70% of the data was used to train the model.
Test Data: 30% of the data was reserved for testing and validation.

## 3. Tokenization and Padding
To convert text into a numerical format suitable for deep learning, the following steps were taken:

Tokenization: Utilized a tokenizer to convert words into numerical sequences, assigning a unique integer to each word.
Padding: Applied padding to ensure uniform input sequence length, which is necessary for neural network processing.

## 4. Building the Deep Learning Model
For sentiment analysis, I implemented a Recurrent Neural Network (RNN), specifically an LSTM (Long Short-Term Memory) network. The architecture includes:

Embedding Layer: Converts word sequences into dense vector representations.
LSTM Layers: Process sequence data and capture contextual information.
Dense Output Layer: Uses a sigmoid activation function to classify sentiment as positive or negative.

## 5. Training and Evaluation
During model training, I monitored key metrics such as:

## Accuracy: 
Measures the model’s correctness in predicting sentiment.
## Precision and Recall: 
Ensures a balanced prediction performance for positive and negative sentiments.
The model was evaluated on the test set to assess its generalization capabilities and effectiveness on unseen data.

## Conclusion
This project successfully demonstrates the application of deep learning to sentiment analysis. The preprocessing pipeline ensured clean, structured data, and the LSTM model effectively captured text context to classify sentiments accurately. The model is ready for use in various applications, including customer reviews, social media monitoring, and more.

Supplemental Materials
🎥 Video Explanation: https://vimeo.com/1010916935?share=copy
