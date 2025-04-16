import torch
import torchvision.transforms as transforms
from torchvision import models
tfrom PIL import Image
import tkinter as tk
from tkinter import filedialog, Label, Button

model = models.resnet18(pretrained=True)
model.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

def classify_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = model(batch_t)
    _, index = torch.max(out, 1)
    return labels[index[0]]

def open_image():
    path = filedialog.askopenfilename()
    if path:
        prediction = classify_image(path)
        result_label.config(text=prediction)

def create_ui():
    root = tk.Tk()
    root.title('Image Classifier AI')
    root.geometry('300x150')
    Button(root, text='Select Image', command=open_image).pack(pady=10)
    global result_label
    result_label = Label(root, text='Prediction will appear here')
    result_label.pack()
    root.mainloop()

create_ui()
