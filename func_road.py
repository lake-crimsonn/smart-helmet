from PIL import Image
import torch
from torchvision import transforms
import time
import cv2

device = 'cpu'
model = torch.load('model.pt', map_location=device)
model.eval()

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

name = ['Not road','Road']

def road(image):
    y, x = image.shape[0], image.shape[1]
    image = Image.fromarray(image)
    image = image.crop((2 * x // 7, y // 3, 5 * x // 7, y))
    image = image.convert('RGB')
    image = transform_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return name[preds[0].int()]
        #print(name[preds[0].int()])
