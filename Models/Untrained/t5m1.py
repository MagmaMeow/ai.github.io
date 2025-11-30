import re
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

CONCORD_NAME = "Concord Integration"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True

EPOCHS = 250000
LR = 0.001
LOG_INTERVAL = 10000
TYPE_SPEED = 0.01

if PRINT_DEVICE:
    print(f"{CONCORD_NAME} starting on device: {DEVICE}")

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

train_data = [
    ("I love AI", 1), ("I enjoy music", 1), ("I like pizza", 1),
    ("I love coding projects", 1), ("I enjoy school when lessons are fun", 1),
    ("I love my friends and my class", 1), ("I enjoy sports", 1),
    ("I love sunny days", 1), ("I enjoy nature walks", 1),
    ("I hate homework", 0), ("I dislike school rules", 0),
    ("I hate slow internet", 0), ("I dislike loud music", 0),
    ("I hate exams", 0), ("I dislike boring lectures", 0),
    ("I hate confusion", 0), ("I dislike crowded places", 0)
]

def build_vocab(pairs):
    vocab = {}
    for sentence, _ in pairs:
        for word in normalize(sentence).split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

word_to_idx = build_vocab(train_data)

def vectorize(sentence, vocab):
    sentence = normalize(sentence)
    vec = [0] * len(vocab)
    for word in sentence.split():
        if word in vocab:
            vec[vocab[word]] += 1
    return torch.tensor(vec, dtype=torch.float32)

X = torch.stack([vectorize(s, word_to_idx) for s, _ in train_data]).to(DEVICE)
y = torch.tensor([label for _, label in train_data], dtype=torch.float32).to(DEVICE)

class TinyConcord(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = TinyConcord(input_dim=X.shape[1]).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % LOG_INTERVAL == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:6d} | Loss: {loss.item():.6f}")

model.eval()

TOPIC_KEYS = [
    "ai","homework","school","music","pizza","coding","friends","exams",
    "internet","games","books","team","rules","bugs","library","math",
    "robots","stories","football","family","nature"
]

def extract_topics(text):
    words = normalize(text).split()
    return [k for k in TOPIC_KEYS if k in words]

def memory_summary(memory):
    if not memory["topics"]:
        return None
    top = sorted(memory["topics"].items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, v in top[:3]]

def update_memory(memory, user_text, assistant_text, prob):
    for t in extract_topics(user_text):
        memory["topics"][t] = memory["topics"].get(t, 0) + 1
    memory["last_prob"] = prob
    memory["turns"] += 1
    memory["history"].append(("user", user_text))
    memory["history"].append(("assistant", assistant_text))
    if len(memory["history"]) > 40:
        memory["history"] = memory["history"][-40:]

# --------------------------
# NEW CHATBOT STYLE REPLY
# --------------------------
def chatbot_reply(user_text, prob, memory):
    tone = "neutral"
    if prob > 0.65:
        tone = "positive"
    elif prob < 0.35:
        tone = "negative"

    lead_ins = {
        "positive": [
            "That sounds genuinely uplifting.",
            "I like the energy behind that.",
            "That brightens the path forward."
        ],
        "negative": [
            "That weighs differently — I can sense it.",
            "That feels heavy on the heart.",
            "The frustration there is real."
        ],
        "neutral": [
            "Interesting perspective.",
            "I’m processing that with you.",
            "There’s something thoughtful in that."
        ]
    }

    connectors = [
        "Tell me more.",
        "How does that impact you?",
        "What’s the deeper story behind it?",
        "What changed recently?"
    ]

    topics = extract_topics(user_text)
    topic_part = ""
    if topics:
        topic_part = f"I’m hearing {', '.join(topics)}. "

    history_hint = ""
    top_topics = memory_summary(memory)
    if top_topics:
        history_hint = f"Earlier you mentioned {', '.join(top_topics)} — how does this connect? "

    reply = (
        f"{CONCORD_NAME}: "
        f"{random.choice(lead_ins[tone])} "
        f"{topic_part}{history_hint}"
        f"{random.choice(connectors)}"
    )

    return reply

def predict_prob(sentence):
    with torch.no_grad():
        x = vectorize(sentence, word_to_idx).to(DEVICE).unsqueeze(0)
        p = model(x).item()
        return p

def typing_print(text):
    if TYPE_SPEED <= 0:
        print(text)
        return
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(TYPE_SPEED)
    print()

HELP_TEXT = (
    f"{CONCORD_NAME}: Commands: /help, /summary, /topics, /reset, /quit\n"
)

def cmd_summary(memory):
    top = memory_summary(memory)
    if not top:
        return f"{CONCORD_NAME}: I’m still forming an internal map of your priorities."
    return f"{CONCORD_NAME}: Key themes emerging: {', '.join(top)}"

def cmd_topics(memory):
    if not memory["topics"]:
        return f"{CONCORD_NAME}: No dominant topics yet."
    items = sorted(memory["topics"].items(), key=lambda kv: kv[1], reverse=True)
    topk = [f"{k}({v})" for k, v in items[:8]]
    return f"{CONCORD_NAME}: Strong repeating topics: {', '.join(topk)}"

def main():
    memory = {"topics": {}, "history": [], "turns": 0, "last_prob": None}
    print("\n--- Concord Integration is active ---")
    typing_print("Say something to begin.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        cmd = user_text.lower()

        if cmd == "/quit":
            typing_print(f"{CONCORD_NAME}: Until next time — stay sharp.")
            break
        if cmd == "/help":
            typing_print(HELP_TEXT)
            continue
        if cmd == "/reset":
            memory = {"topics": {}, "history": [], "turns": 0, "last_prob": None}
            typing_print(f"{CONCORD_NAME}: Memory wiped. Fresh page.")
            continue
        if cmd == "/summary":
            typing_print(cmd_summary(memory))
            continue
        if cmd == "/topics":
            typing_print(cmd_topics(memory))
            continue

        prob = predict_prob(user_text)
        reply = chatbot_reply(user_text, prob, memory)
        typing_print(reply)
        update_memory(memory, user_text, reply, prob)

if __name__ == "__main__":
    main()
