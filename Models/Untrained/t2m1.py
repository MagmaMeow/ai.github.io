# Concord Integration — lightweight conversational sentiment micro-model
# Author: Samridh
# Goal: Listen to user input and answer naturally with confidence-aware tone

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 0) Persona and settings
# -----------------------------
CONCORD_NAME = "Concord Integration"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True  # set False if you want silent device selection
EPOCHS = 50000        # you can push higher (e.g., 20_000) for near-0 loss
LR = 0.01            # fine-tuning friendly learning rate

if PRINT_DEVICE:
    print(f"{CONCORD_NAME} starting on device: {DEVICE}")

# -----------------------------
# 1) Training data (human-style sentences)
# Balanced positives and negatives with varied topics
# -----------------------------
train_data = [
    # Positive
    ("I love AI", 1),
    ("I enjoy music", 1),
    ("I like pizza", 1),
    ("I love homework sometimes", 1),
    ("I enjoy school when lessons are fun", 1),
    ("I like learning new things", 1),
    ("I love playing games with friends", 1),
    ("I enjoy coding projects", 1),
    ("I like reading books at night", 1),
    ("I love my friends and my class", 1),

    # Negative
    ("I hate homework", 0),
    ("I dislike school rules", 0),
    ("I hate AI sometimes", 0),
    ("I dislike pizza today", 0),
    ("I hate loud music", 0),
    ("I dislike learning under pressure", 0),
    ("I hate playing games alone", 0),
    ("I dislike coding bugs", 0),
    ("I hate reading boring books", 0),
    ("I dislike my friends being rude", 0),
]

# -----------------------------
# 2) Build vocabulary dynamically from training sentences
# -----------------------------
def build_vocab(pairs):
    vocab = {}
    for sentence, _ in pairs:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

word_to_idx = build_vocab(train_data)

# -----------------------------
# 3) Vectorizer (bag-of-words)
# -----------------------------
def vectorize(sentence, vocab):
    vec = [0] * len(vocab)
    for word in sentence.split():
        if word in vocab:
            vec[vocab[word]] += 1
        # if a word is unseen, we ignore it in this simple BoW setup
    return torch.tensor(vec, dtype=torch.float32)

# -----------------------------
# 4) Create tensors
# -----------------------------
X = torch.stack([vectorize(s, word_to_idx) for s, _ in train_data]).to(DEVICE)
y = torch.tensor([label for _, label in train_data], dtype=torch.float32).to(DEVICE)

# -----------------------------
# 5) Model: simple linear + sigmoid
# -----------------------------
class TinyConcord(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Using sigmoid for probability
        return torch.sigmoid(self.fc(x))

model = TinyConcord(input_dim=X.shape[1]).to(DEVICE)

# -----------------------------
# 6) Loss and optimizer
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# -----------------------------
# 7) Training loop
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Light logging every 200 epochs
    if epoch % 200 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

# -----------------------------
# 8) Concord-style reply generator
# -----------------------------
def concord_reply(prob, sentence):
    """
    prob: model's positive sentiment probability in [0,1]
    Returns a natural language response with confidence-aware tone.
    """
    # Confidence thresholds
    high = 0.85
    low = 0.15
    unsure_lo = 0.45
    unsure_hi = 0.55

    # Uncertainty band
    if unsure_lo <= prob <= unsure_hi:
        return f"{CONCORD_NAME}: I’m a bit unsure about how that feels. Tell me more about why you said: “{sentence}”? (confidence {prob:.4f})"

    # Strong positive
    if prob >= high:
        return f"{CONCORD_NAME}: That sounds genuinely uplifting. I hear optimism in “{sentence}”. (confidence {prob:.4f})"

    # Strong negative
    if prob <= low:
        return f"{CONCORD_NAME}: That feels heavy. I hear frustration in “{sentence}”. (confidence {prob:.4f})"

    # Mild positive vs mild negative
    if prob > 0.5:
        return f"{CONCORD_NAME}: It leans positive—curious what’s making it feel good in “{sentence}”. (confidence {prob:.4f})"
    else:
        return f"{CONCORD_NAME}: It leans negative—what’s behind the tension in “{sentence}”? (confidence {prob:.4f})"

# -----------------------------
# 9) Interactive loop (listen and answer)
# -----------------------------
def predict_sentence(sentence):
    with torch.no_grad():
        x = vectorize(sentence, word_to_idx).to(DEVICE).unsqueeze(0)
        p = model(x).item()
        return p

print("\n--- Concord Integration is ready. Type your sentence and press Enter ---")
print("Type '/quit' to stop. Type '/help' for tips.\n")

while True:
    user_inp = input("You: ").strip()
    if not user_inp:
        continue
    if user_inp.lower() == "/quit":
        print(f"{CONCORD_NAME}: Talk soon. Keep creating.")
        break
    if user_inp.lower() == "/help":
        print(f"{CONCORD_NAME}: I understand everyday phrases. I react to words like love, enjoy, like, hate, dislike, homework, school, music, pizza, coding, friends.")
        print(f"{CONCORD_NAME}: If I seem unsure, try rephrasing or add detail. I’m a tiny model—concrete words help.")
        continue

    prob = predict_sentence(user_inp)
    print(concord_reply(prob, user_inp))
