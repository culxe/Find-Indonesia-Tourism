# **Find Indonesia Tourism - Machine Learning**

This project focuses on building a Natural Language Processing (NLP) sentiment analysis model using TensorFlow. The objective is to analyze reviews from Google Maps to determine whether they are positive or negative, after that, we used this model to make a search box to find very positive attraction around. The project is structured as follows:

## Collect Dataset from Google Maps Reviews
The dataset consists of over 2800 Google Maps reviews related to various tourist destinations in Indonesia. These reviews are labeled with target values: 0 for negative sentiments and 1 for positive sentiments. The dataset can be accessed [here](https://github.com/BangkitCapstoneFIT/ML-findindonesiatourism/blob/main/Book133333.xlsb.csv)

<img src="/img/3.info-dataset(2).png" alt="Alt text" width="800"/>

with detail info dataset

<img src="/img/2.info-dataset.png" alt="Alt text" width="800"/>

## Data Pre-Processing
Before training the model, the text data undergoes several preprocessing steps:
- Regular Expressions (Regex): Used to clean the text by removing unwanted characters, special symbols, and emojis that do not contribute to the sentiment.
- Stopwords Removal: Commonly used words that do not add much meaning to the text (e.g., "is", "the", "and") are removed to reduce noise and improve model performance.
- 
processed data

<img src="/img/4.processed-data.png" alt="Alt text" width="800"/>

## Word Embedding
To convert the text into a format that the machine learning model can understand:

- Tokenizer: The text is tokenized, meaning it is split into individual words or tokens.
- GloVe Word Embeddings: The GloVe (Global Vectors for Word Representation) embeddings are used to convert these tokens into dense vectors. This helps the model understand the context and relationships between words.

## Model Training
The preprocessed and embedded data is used to train a Convolutional Neural Network (CNN) model. The model's architecture includes:

- Convolutional 1D Layers: Used to capture spatial hierarchies in the text data.
- Activation Function: ReLU (Rectified Linear Unit) is used for non-linear transformation, and - Sigmoid is used in the final output layer for binary classification.
- Optimizer: Adam optimizer is used to minimize the loss function.
- Loss Function: Binary Crossentropy is chosen as the loss function because this is a binary classification problem.
- 
<img src="/img/5.conf-model.png" alt="Alt text" width="800"/>


## Model Evaluation and Prediction
The model is evaluated on the test set, achieving an accuracy of 79% and a loss of 0.58. The model's prediction on a sample data results in a value of 0.99406624, indicating a strong positive sentiment.
<img src="/img/6.model-acc.png" alt="Alt text" width="800"/> <img src="/img/7.model-loss.png" alt="Alt text" width="800"/>

predicted result

 <img src="/img/1.predict.png" alt="Alt text" width="800"/>

##  Deployment
The trained model is converted to TensorFlow Lite format for deployment on Android devices. This allows for on-device sentiment analysis, enabling users to quickly assess the sentiment of reviews directly on their mobile devices.

For more detailed steps and implementation, you can refer to the [notebook](https://github.com/BangkitCapstoneFIT/ML-findindonesiatourism/blob/main/Analysis_Sentimen_Find_Indonesia_Tourism.ipynb)
