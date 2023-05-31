# TPDL



**Prerequisites**
Before running the Bert code, make sure you have the following dependencies installed:

pandas
numpy
torch
scikit-learn
transformers

You can install these dependencies using pip:
**pip install pandas numpy torch scikit-learn transformers**


Before running the LSTM code, make sure you have the following dependencies installed:

pandas
numpy
openpyxl
scikit-learn
keras
tensorflow
You can install these dependencies using pip:
**pip install pandas numpy openpyxl scikit-learn keras tensorflow**


**BERT**
**Model Training**
The BERT model is fine-tuned for sequence classification using the training dataset. The citation context text is tokenized using the BERT tokenizer, and the labels are converted to numerical values. The processed data is then used to create PyTorch datasets and dataloaders for efficient training.

The pre-trained BERT model for sequence classification (bert-base-uncased) is loaded, and an AdamW optimizer with a learning rate of 2e-5 is set up. A learning rate scheduler is also implemented to adjust the learning rate during training.

The model is trained for a specified number of epochs, and the loss is printed for each epoch. At the end of training, the model is saved.

**Model Evaluation**
After training, the model is evaluated on the test set. The model is switched to evaluation mode, and the test data is passed through the model to obtain predictions. The loss, accuracy, classification report, and confusion matrix are calculated and printed.

The classification report provides precision, recall, F1-score, and support for each class (No and Yes). The confusion matrix displays the true positive, true negative, false positive, and false negative values.


**LSTM**
**Model Training**
The citation context text is tokenized using the Keras Tokenizer, and the labels are encoded as numerical values using the LabelEncoder from scikit-learn. The tokenized sequences are padded to a maximum length to ensure uniform input size.

An LSTM-based neural network architecture is defined using the Keras Sequential model. The architecture consists of an Embedding layer, LSTM layer with dropout, and a Dense output layer with sigmoid activation for binary classification.

The model is compiled with the binary cross-entropy loss function, the Adam optimizer, and accuracy as the evaluation metric. The model is then trained on the training dataset for a specified number of epochs and batch size.

**Model Evaluation**
After training, the model is evaluated on the test set. The model makes predictions on the test sequences, and a threshold of 0.5 is used to classify the predictions as either Non-Dependent or Dependent. The classification report, including precision, recall, F1-score, and support, is printed. The confusion matrix is also displayed, showing true positive, true negative, false positive, and false negative values.


**Note:** This code assumes the availability of a GPU for training the BERT model. If a GPU is not available, the code will automatically use the CPU, but training may be slower.


**Machine Learning Classifiers**
**Prerequisites**
Before running the code, make sure you have the following dependencies installed:

pandas
scikit-learn
You can install these dependencies using pip:
**pip install pandas scikit-learn**
**Feature Extraction**
The citation context text is converted into bi-gram vectors using the CountVectorizer from scikit-learn. This vectorizer creates a vocabulary of bi-grams and represents each citation context as a binary vector indicating the presence or absence of each bi-gram in the context.

**Model Training and Evaluation**
The data is split into training and testing sets using the train_test_split function from scikit-learn. The training set is used to train each classifier, and the testing set is used for evaluation.

Three classifiers are trained and evaluated:

**Support Vector Machine (SVM) with a linear kernel.
Logistic Regression.
Naive Bayes classifier.**

For each classifier, the training set is used to fit the model, and then predictions are made on the testing set. The accuracy of each classifier is calculated using the accuracy_score function from scikit-learn. Additionally, a classification report is printed, which includes precision, recall, F1-score, and support for each class.
