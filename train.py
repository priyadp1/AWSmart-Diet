import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from classifier import BERTClassifier

class NutritionData(Dataset):
    def __init__(self, texts, labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self , idx):
        inputs = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, label
    
def train(model, dataloader, device, epochs=3):
    optimizer = torch.optim.Adam(model.parameters() , lr=2e-5)
    criterion = torch.nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item()}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_file', type=str, default='/opt/ml/input/data/train/nutrition_cleaned_recovered.csv')

    args = parser.parse_args()

    df = pd.read_csv(args.train_file)
    labels = ['iron', 'vitamin_c', 'calcium', 'protein']
    df = df.dropna(subset=['text'] + labels)
    X = df['text'].tolist()
    y = df[labels].astype(int).values.tolist()

    dataset = NutritionData(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier().to(device)

    train(model, dataloader, device, epochs=args.epochs)

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__ == '__main__':
    main()
