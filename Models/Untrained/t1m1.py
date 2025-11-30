import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Device selection (GPU if available, else CPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. Dummy dataset (Bag of Words style)
# -----------------------------
# Vocabulary: [I, love, AI, hate, homework]
word_to_idx = {"I":0, "love":1, "AI":2, "hate":3, "homework":4}

X = torch.tensor([[1,1,1,0,0],   # "I love AI" -> Positive
                  [1,0,1,1,0],   # "I hate AI" -> Negative
                  [1,1,0,0,1],   # "I love homework" -> Positive
                  [1,0,0,1,1]],  # "I hate homework" -> Negative
                 dtype=torch.float32)

y = torch.tensor([1,0,1,0], dtype=torch.float32)  # Labels

X, y = X.to(device), y.to(device)

# -----------------------------
# 3. Define simple model
# -----------------------------
class SimpleNLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNLP, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = SimpleNLP(input_dim=X.shape[1]).to(device)

# -----------------------------
# 4. Training setup
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# -----------------------------
# 5. Training loop
# -----------------------------
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -----------------------------
# 6. Ask user for sentences
# -----------------------------
def vectorize(sentence):
    vec = [0]*len(word_to_idx)
    for word in sentence.split():
        if word in word_to_idx:
            vec[word_to_idx[word]] += 1
    return torch.tensor(vec, dtype=torch.float32)

s1 = input("Enter first sentence: ")
s2 = input("Enter second sentence: ")

test = torch.stack([vectorize(s1), vectorize(s2)]).to(device)

# -----------------------------
# 7. Predictions
# -----------------------------
with torch.no_grad():
    preds = model(test).squeeze()
    print("Predictions:", preds.cpu().numpy())
    for i, p in enumerate(preds.cpu().numpy()):
        label = "Positive" if p >= 0.5 else "Negative"
        print(f"Sentence {i+1}: {label} ({p:.4f})")
