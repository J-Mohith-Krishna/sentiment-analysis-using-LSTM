import re 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from keras.preprocessing.text import Tokenizer 
from keras_preprocessing.sequence import pad_sequences 
import keras 
from sklearn.metrics import classification_report, accuracy_score 
import nltk 

data = pd.read_csv('IMDB Dataset.csv') 

def remove_tags(string): 
    removelist = "" 
    result = re.sub('','',string)
    result = re.sub('https://.*','',result)
    result = re.sub(r'[^w'+removelist+']', ' ',result)
    result = result.lower() 
    return result 

data['review'] = data['review'].apply(lambda cw : remove_tags(cw)) 

nltk.download('stopwords') 
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])) 

w_tokenizer = nltk.tokenize.WhitespaceTokenizer() 
lemmatizer = nltk.stem.WordNetLemmatizer() 

def lemmatize_text(text): 
    st = "" 
    for w in w_tokenizer.tokenize(text): 
        st = st + lemmatizer.lemmatize(w) + " " 
    return st 

data['review'] = data.review.apply(lemmatize_text) 

s = 0.0 
for i in data['review']: 
    word_list = i.split() 
    s = s + len(word_list) 
print("Average length of each review : ",s/data.shape[0]) 

pos = 0 
for i in range(data.shape[0]): 
    if data.iloc[i]['sentiment'] == 'positive': 
        pos = pos + 1 

neg = data.shape[0] - pos 
print("Percentage of reviews with positive sentiment is "+str(pos/data.shape[0]*100)+"%") 
print("Percentage of reviews with negative sentiment is "+str(neg/data.shape[0]*100)+"%") 

reviews = data['review'].values 
labels = data['sentiment'].values 
encoder = LabelEncoder() 
encoded_labels = encoder.fit_transform(labels) 

train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels, stratify=encoded_labels) 

vocab_size = 3000
oov_tok = '' 
embedding_dim = 100 
max_length = 200 
padding_type='post' 
trunc_type='post' 

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok) 
tokenizer.fit_on_texts(train_sentences) 
word_index = tokenizer.word_index 

train_sequences = tokenizer.texts_to_sequences(train_sentences) 
train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length) 

test_sequences = tokenizer.texts_to_sequences(test_sentences) 
test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length) 

model = keras.Sequential([ 
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), 
    keras.layers.Bidirectional(keras.layers.LSTM(64)), 
    keras.layers.Dense(24, activation='relu'), 
    keras.layers.Dense(1, activation='sigmoid') 
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

num_epochs = 5 
history = model.fit(train_padded, train_labels, epochs=num_epochs, verbose=1, validation_split=0.1) 

prediction = model.predict(test_padded) 

pred_labels = [] 
for i in prediction: 
    if i >= 0.5: 
        pred_labels.append(1) 
    else: 
        pred_labels.append(0) 

print("Accuracy of prediction on test set : ", accuracy_score(test_labels, pred_labels)) 

sentence = ["The movie was very touching and heart whelming", 
            "I have never seen a terrible movie like this", 
            "the movie plot is terrible but it had good acting"] 

sequences = tokenizer.texts_to_sequences(sentence) 
padded = pad_sequences(sequences, padding='post', maxlen=max_length) 

prediction = model.predict(padded) 

pred_labels = [] 
for i in prediction: 
    if i >= 0.5: 
        pred_labels.append(1) 
    else: 
        pred_labels.append(0) 

for i in range(len(sentence)): 
    print(sentence[i]) 
    if pred_labels[i] == 1: 
        s = 'Positive' 
    else: 
        s = 'Negative' 
    print("Predicted sentiment :", s)
