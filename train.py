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

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, label

def train(model, dataloader, device, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './models'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './'))

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

   
    csv_path = os.path.join(args.train, "nutrition_cleaned_recovered.csv")
    df = pd.read_csv(csv_path)

    labels = ['iron', 'vitamin_c', 'calcium', 'protein']
    df = df.dropna(subset=['description'] + labels)

    X = df['description'].tolist()
    y = df[labels].clip(upper=1).astype(int).values.tolist()

    dataset = NutritionData(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier().to(device)

    train(model, dataloader, device, epochs=args.epochs)

    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
