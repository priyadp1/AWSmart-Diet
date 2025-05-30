import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NutritionClassifier(nn.Module):
    def __init__(self):
        super(NutritionClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)

def loadModel(path="model.pth"):
    model = NutritionClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def classify(meal, model):
    features = ['iron', 'vitamin_c', 'calcium', 'protein']
    try:
        inputTensor = torch.tensor([float(meal[f]) for f in features], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(inputTensor)
            _, predicted = torch.max(output, 1)
            return "good" if predicted.item() == 1 else "not ideal"
    except:
        return "N/A"


class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, num_classes)  # 4 = number of deficiency classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:,0,:]
        return self.fc(pooled_output)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BERTClassifier(num_classes=4)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

label_map = {
    0: "iron",
    1: "calcium",
    2: "vitamin D",
    3: "vitamin C"
}

def predict_deficiency(text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids=input_ids , attention_mask=attention_mask)
        prediction = torch.argmax(logits, dim=1).item()

    return label_map.get(prediction, "unknown")
