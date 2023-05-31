import pandas as pd
import numpy as np
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix, classification_report

# Load data from CSV file
import pandas as pd

# Read Excel file into a DataFrame
data = pd.read_excel('Annotation-MU.xlsx')

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Encode dependent and non-dependent labels
le = LabelEncoder()
train_labels = le.fit_transform(train_data['Class'])
test_labels = le.transform(test_data['Class'])

# Tokenize citation context text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['CitationContext'])
train_sequences = tokenizer.texts_to_sequences(train_data['CitationContext'])
test_sequences = tokenizer.texts_to_sequences(test_data['CitationContext'])

# Pad sequences to same length
max_len = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_len)
test_sequences = pad_sequences(test_sequences, maxlen=max_len)

# Define neural network architecture
model = Sequential()
model.add(Embedding(5000, 128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(train_sequences, train_labels, epochs=10, batch_size=32, validation_data=(test_sequences, test_labels))

# Make predictions on test set
y_pred = model.predict(test_sequences)
y_pred = (y_pred > 0.5)

# Print classification report
print(classification_report(test_labels, y_pred, target_names=['Non-Dependent', 'Dependent']))

# Print confusion matrix
print(confusion_matrix(test_labels, y_pred))

