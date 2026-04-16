import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import time

def test_optimizer(opt_class, name, **kwargs):
    print(f"\nTesting {name}...")
    try:
        device = torch_directml.device()
        model = nn.Linear(10, 1).to(device)
        optimizer = opt_class(model.parameters(), **kwargs)
        
        data = torch.randn(64, 10).to(device)
        target = torch.randn(64, 1).to(device)
        
        start = time.time()
        for _ in range(10):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()
        end = time.time()
        print(f"{name} finished in {end - start:.4f}s")
        
    except Exception as e:
        print(f"{name} failed: {e}")

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"DirectML device: {torch_directml.device()}")
    
    test_optimizer(optim.Adam, "Adam", lr=0.001)
    test_optimizer(optim.AdamW, "AdamW", lr=0.001)
    test_optimizer(optim.RMSprop, "RMSprop", lr=0.001)
    test_optimizer(optim.SGD, "SGD", lr=0.001, momentum=0.9)
