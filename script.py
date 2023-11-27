import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

classes = ['dahlia', 'jasmine', 'lilac', 'lotus', 'protea', 'violet', 'zinnia']


class ImagePredictor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model_path)
        self.transform = self.load_transform()

    def load_model(self, model_path):
        model = models.resnet50(pretrained=True).to(self.device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(classes)).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def load_transform(self):
        img_height = 256
        img_width = 256
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def get_image_from_url(self, image_url):
        import requests
        from PIL import Image
        from io import BytesIO

        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image

    def predict_image(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        output = self.model(image)
        _, predicted = torch.max(output, 1)
        prob = F.softmax(output, dim=1)[0] * 100
        prob_res = round(prob[predicted[0]].item(), 2)

        return f'<span style="color: blue">Predicted: {classes[predicted[0]]}</span><br/>' \
               f'Percent: {prob_res}'