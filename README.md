# **Find Indonesia Tourism - Machine Learning**

Natural Language Processing Sentiment Analysis model with Tensorflow. Data collected from Google Maps Reviews.

### Collect Dataset from Google Maps Reviews
Collected 2800+ [dataset](https://github.com/BangkitCapstoneFIT/ML-findindonesiatourism/blob/main/Book133333.xlsb.csv), this dataset contain text and target (0 negative or 1 positive). 
### Data Pre-Processing
Using regular expression and stopword to remove unnecessary characters and emoji
### Embedding Words
Embedding words with Tokenizer and Glove word embeddings.
### Model Training
Model training with CNN Model, Convolution 1D, ReLu, Sigmoid, optimizer Adam, and Binary Crossentropy.
### Saved Models and Predict
Model performance with 79% accuracy and loss 0.58. Model Predict result is a 0.99406624. Deployed to android with Tensorflow lite. The notebook refers to this [link](https://github.com/BangkitCapstoneFIT/ML-findindonesiatourism/blob/main/Analysis_Sentimen_Find_Indonesia_Tourism.ipynb)
