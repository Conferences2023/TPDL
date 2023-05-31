import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

# Load data from Excel file
data = pd.read_excel('Annotation-MU.xlsx')

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenize citation context text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_data['CitationContext']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data['CitationContext']), truncation=True, padding=True)

# Convert labels to numerical values
label_map = {"Yes": 1, "No": 0}
train_labels = train_data['RetractionMentioned'].apply(lambda x: label_map[x])
test_labels = test_data['RetractionMentioned'].apply(lambda x: label_map[x])

# Create PyTorch datasets
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), torch.tensor(train_labels.values))
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']), torch.tensor(test_labels.values))

# Create PyTorch dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

# Train model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

for epoch in range(5):
    running_loss = 0.0
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader)}")

# Evaluate model on test set
model.eval()
test_loss, test_acc, test_predictions, test_targets = 0, 0, [], []
with torch.no_grad():
    for batch in test_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss, logits = outputs.loss, outputs.logits
        test_loss += loss.item()
        test_predictions.extend(torch.argmax(logits, axis=1).tolist())
        test_targets.extend(inputs['labels'].tolist())
test_loss /= len(test_loader)
test_acc = sum(np.array(test_predictions) == np.array(test_targets)) / len(test_targets)

# Print classification report and confusion matrix
print(classification_report(test_targets, test_predictions,target_names=['No', 'Yes']))

# Print confusion matrix
print(confusion_matrix(test_labels, predictions))


