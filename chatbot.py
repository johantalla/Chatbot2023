import pandas as pd
import keras
import tensorflow as tf
from keras.layers import TextVectorization
import json
from sklearn.model_selection import train_test_split
import random
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

train_df = pd.read_csv('Mental_Health_FAQ.csv')
train_df_shuffled = train_df.sample(frac=1, random_state=7)
train_df_shuffled.head()


# Use train_test_split to split training data into training and test sets
train_df, test_df, train_labels, test_labels = train_test_split(train_df_shuffled["Questions"].to_numpy(),
                                                                            train_df_shuffled["Answers"].to_numpy(),
                                                                            test_size=0.1, # 10% of samples for validation set
                                                                            random_state=42) # random state to allow for reproducibility
text_vectorizer = TextVectorization(max_tokens=30000, # We do not know how many different words there are
                                    standardize="lower_and_strip_punctuation", # how to process the input texts
                                    split="whitespace", # Splits to tokens on empty spaces
                                    ngrams=None,
                                    output_mode="int", # how to map tokens to numbers
                                    output_sequence_length=20) # how long should the output sequence of tokens be?
                                    # pad_to_max_tokens=True) # Not valid if using max_tokens=None
text_vectorizer.adapt(train_df)
random_sentence = random.choice(train_df)
sent='Hello my name is johan and im here to help today'
print(random_sentence)
print(f'original sentence: {random_sentence}\ntokenized version:{text_vectorizer([sent])}')
words_in_vocab = text_vectorizer.get_vocabulary()
'''print(words_in_vocab[-5:])
print(words_in_vocab[:5])'''

#CREATING AN EMBEDDING
'''embedding = layers.Embedding(input_dim=len(text_vectorizer.get_vocabulary()),
                             output_dim=128,
                             embeddings_initializer='uniform',
                             name='embedding1')
train_df_embed = embedding(train_df)
train_label_embed = embedding(train_labels)'''
model_0 = Pipeline([
                    ("tfidf", TfidfVectorizer()), # convert words to numbers using tfidf
                    ("clf", MultinomialNB()) # model the text
])

# Fit the pipeline to the training data
'''model_0.fit(train_df, train_labels)
baseline_score = model_0.score(test_df, test_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")
baseline_preds = model_0.predict(test_df)
print(baseline_preds[:20])'''