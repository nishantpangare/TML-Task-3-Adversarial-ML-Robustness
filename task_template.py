import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet50
import torch.optim as optim


# Hyperparameters
data_path = "/home/atml_team013/tml3/train.npz"
val_size = 5000 
batch_size = 128
NUM_CLASSES = 9
epochs = 120
lr = 0.1  
eps = 8 / 255 
alpha = 2 / 255     
pgd_train_steps = 20
pgd_val_steps = 20     


save_best = "model20.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# the dataset is provided as a .npz file (compressed numpy archive)
# it contains two arrays:
# images: uint8 array of shape (N, 3, 32, 32), values in [0, 255]
# labels: integer class labels in range [0, 8]
# we divide images by 255.0 to get float values in [0, 1]
def load_data():
    data  = np.load(data_path)
    images = torch.from_numpy(data["images"]).float() / 255.0
    labels = torch.from_numpy(data["labels"]).long()

    print("Dataset size:", len(images))
    print("Image shape:", images.shape)
    print("Label range:", labels.min().item(), "to", labels.max().item())

    full_data = TensorDataset(images, labels)
    train_size = len(full_data) - val_size 
    train_ds, val_ds = random_split(full_data, [train_size, val_size],generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size=256,shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def build_model():
    model = resnet50(weights=None)               
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    #sanity check - output shape must be (1, 9)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32).to(DEVICE))
    print(f"Output shape: {out.shape}")

    return model


def pgd(model,imgs,labels,eps,alpha,steps):
    
    imgs = imgs.clone().detach().to(DEVICE)
    labels = labels.clone().detach().to(DEVICE)

    delta = torch.empty_like(imgs).uniform_(-eps,eps)
    delta = torch.clamp(imgs + delta,0, 1) - imgs

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(imgs+delta), labels)
        loss.backward()
        grad = delta.grad.detach()
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(imgs + delta, 0, 1) - imgs
    return (imgs + delta).detach()


@torch.no_grad()
def clean_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE),labels.to(DEVICE)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


def robust_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        adv = pgd(model, imgs, labels,eps=eps, alpha=alpha, steps=pgd_val_steps)
        with torch.no_grad():
            correct += model(adv).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


def train():
    best_score = 0.0
    train_loader , val_loader = load_data()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr = lr, momentum=0.9,weight_decay=5e-4,nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0.0, 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            adv_imgs = pgd(model,imgs,labels, eps=eps,alpha=alpha, steps=pgd_train_steps)

            model.train()
            optimizer.zero_grad()
            loss = 0.5 * criterion(model(imgs),labels) + 0.5 * criterion(model(adv_imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)

        scheduler.step()

        clean_acc = clean_accuracy(model,val_loader)
        robust_acc = robust_accuracy(model,val_loader)
        score = 0.5 * clean_acc + 0.5 * robust_acc

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(),save_best)

        
        print(f"\nEpochs: {epoch:3d}/{epochs} || Loss: {total_loss/len(train_loader):.2f}")
        print(f"Clean Accuracy: {clean_acc:.3f} || Robust Accuracy: {robust_acc:.3f} || Final Score: {score:.4f}")      

    print(f"Best Score: {best_score:.4f}")

if __name__ == "__main__":
    train()