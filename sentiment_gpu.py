import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Device selection (any GPU or CPU)
# -----------------------------
if torch.cuda.is_available():
    print("GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda")  # automatically picks the first GPU
else:
    print("No GPU found, using CPU.")
    device = torch.device("cpu")

print("Using device:", device)

# -----------------------------
# 2. Dummy dataset (Bag of Words style)
# -----------------------------
# Features: [I, love, AI, hate, homework]
X = torch.tensor([[1,1,1,0,0],   # "I love AI" -> Positive
                  [1,0,1,1,0],   # "I hate AI" -> Negative
                  [1,1,0,0,1],   # "I love homework" -> Positive
                  [1,0,0,1,1]],  # "I hate homework" -> Negative
                 dtype=torch.float32)

y = torch.tensor([1,0,1,0], dtype=torch.float32)  # Labels: 1=Positive, 0=Negative

# Move data to chosen device
X, y = X.to(device), y.to(device)

# -----------------------------
# 3. Define simple model
# -----------------------------
class SimpleNLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNLP, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # one output neuron

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = SimpleNLP(input_dim=X.shape[1]).to(device)

# -----------------------------
# 4. Training setup
# -----------------------------
criterion = nn.BCELoss()  # binary cross-entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)

# -----------------------------
# 5. Training loop
# -----------------------------
for epoch in range(27000):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -----------------------------
# 6. Test predictions
# -----------------------------
with torch.no_grad():
    test = torch.tensor([[1,1,1,0,0],   # "I love AI"
                         [1,0,0,1,1]],  # "I hate homework"
                        dtype=torch.float32).to(device)
    preds = model(test).squeeze()
    print("Predictions:", preds.cpu().numpy())
