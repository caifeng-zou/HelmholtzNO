import torch
device = torch.device('cuda')

def evaluate(model, data_loader):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            out = model(x)
            y_pred.append(out.cpu())
        
    return torch.cat(y_pred, 0)