import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
class NutritionClassifier(nn.Module):
    def __init__(self):
        super(NutritionClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )
    def forward(self , x):
        return self.net(x)

def loadModel(path="model.pth"):
    model = NutritionClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def classify(meal, model):
    features = ['iron' , 'vitamin_c' , 'calcium' , 'protein']
    try:
        inputTensor = torch.tensor([float(meal[f]) for f in features] , dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(inputTensor)
            _, predicted = torch.max(output, 1)
            return "good" if predicted.item() == 1 else "not ideal"
    except:
        return "N/A"
    


class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self , input , mask):
        output = self.bert(input=input, mask=mask)
        pooled = self.drop(output.pooler_output)
        return torch.sigmoid(self.fc(pooled))
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def loadBERT(path="BERTmodel.pth"):
    model = BERTClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(text, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        probs = model(input_ids, attention_mask)
        result = (probs > 0.5).int().squeeze().tolist()
    
    labels = ['iron', 'vitamin_c', 'calcium', 'protein']
    
    if isinstance(result, int):
        result = [result]
    
    return [labels[i] for i, val in enumerate(result) if val == 1]
