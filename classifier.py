import torch
import torch.nn as nn
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