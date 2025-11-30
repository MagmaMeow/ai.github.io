import re
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 0) Settings and persona
# -----------------------------
CONCORD_NAME = "Concord Integration"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True

# Training settings (you can tune)
EPOCHS = 250000
LR = 0.001
LOG_INTERVAL = 10000
TYPE_SPEED = 0.01  # 0 = instant; set e.g., 0.01 to simulate typing

if PRINT_DEVICE:
    print(f"{CONCORD_NAME} starting on device: {DEVICE}")

# -----------------------------
# 1) Text normalization
# -----------------------------
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------------
# 2) Expanded training data (balanced)
#  - Problem definition: binary tone classification (positive vs negative)
#  - Data prep: simple, human-style phrases across domains
# -----------------------------
train_data = [
    # Positive
    ("I love AI", 1), ("I enjoy music", 1), ("I like pizza", 1),
    ("I love homework sometimes", 1), ("I enjoy school when lessons are fun", 1),
    ("I like learning new things", 1), ("I love playing games with friends", 1),
    ("I enjoy coding projects", 1), ("I like reading books at night", 1),
    ("I love my friends and my class", 1), ("I enjoy sports with my team", 1),
    ("I love sunny days", 1), ("I like programming challenges", 1),
    ("I enjoy movies on weekends", 1), ("I love eating ice cream", 1),
    ("I enjoy solving puzzles", 1), ("I like drawing and painting", 1),
    ("I love exploring new places", 1), ("I enjoy talking with my family", 1),
    ("I like playing football", 1), ("I feel excited about my project", 1),
    ("I am hopeful about exams", 1), ("I feel calm before the test", 1),
    ("I enjoy early mornings", 1), ("I like studying with friends", 1),
    ("I love building robots", 1), ("I enjoy writing stories", 1),
    ("I like practicing math", 1), ("I love helping my classmates", 1),
    ("I enjoy a quiet library", 1), ("I like cooking simple meals", 1),
    ("I love learning from mistakes", 1), ("I enjoy thoughtful discussions", 1),
    ("I like staying organized", 1), ("I love clean code", 1),
    ("I enjoy small wins every day", 1), ("I like planning my study schedule", 1),
    ("I love brainstorming ideas", 1), ("I enjoy nature walks", 1),
    ("I like listening to podcasts", 1), ("I love drawing flowcharts", 1),
    ("I enjoy debugging with patience", 1), ("I like trying new activities", 1),
    ("I love teamwork during projects", 1), ("I enjoy sharing knowledge", 1),
    ("I like learning languages", 1), ("I love creative presentations", 1),
    ("I enjoy practicing guitar", 1), ("I like balanced routines", 1),
    ("I love seeing progress", 1),

    # Negative
    ("I hate homework", 0), ("I dislike school rules", 0), ("I hate AI sometimes", 0),
    ("I dislike pizza today", 0), ("I hate loud music", 0),
    ("I dislike learning under pressure", 0), ("I hate playing games alone", 0),
    ("I dislike coding bugs", 0), ("I hate reading boring books", 0),
    ("I dislike my friends being rude", 0), ("I hate exams", 0),
    ("I dislike traffic jams", 0), ("I hate waking up early", 0),
    ("I dislike rainy days", 0), ("I hate losing in games", 0),
    ("I dislike spicy food", 0), ("I hate being alone", 0),
    ("I dislike boring lectures", 0), ("I hate waiting in lines", 0),
    ("I dislike noisy classrooms", 0), ("I feel stressed about projects", 0),
    ("I am frustrated with exams", 0), ("I feel anxious before tests", 0),
    ("I hate messy code", 0), ("I dislike confusing assignments", 0),
    ("I hate slow internet", 0), ("I dislike unfair rules", 0),
    ("I hate running late", 0), ("I dislike crowded places", 0),
    ("I hate long meetings", 0), ("I dislike bitter vegetables", 0),
    ("I hate losing progress", 0), ("I dislike getting distracted", 0),
    ("I hate loud classrooms", 0), ("I dislike stubborn errors", 0),
    ("I hate unclear instructions", 0), ("I dislike cold weather", 0),
    ("I hate failing tests", 0), ("I dislike broken tools", 0),
    ("I hate unnecessary delays", 0), ("I dislike negative attitudes", 0),
    ("I hate careless mistakes", 0), ("I dislike poor planning", 0),
    ("I hate confusion in lessons", 0), ("I dislike boring routines", 0),
    ("I hate wasting time", 0), ("I dislike messy desks", 0),
    ("I hate repetitive tasks", 0), ("I dislike noisy streets", 0),
]

# -----------------------------
# 3) Build vocabulary from normalized data
# -----------------------------
def build_vocab(pairs):
    vocab = {}
    for sentence, _ in pairs:
        for word in normalize(sentence).split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

word_to_idx = build_vocab(train_data)

# -----------------------------
# 4) Bag-of-words vectorizer
# -----------------------------
def vectorize(sentence, vocab):
    sentence = normalize(sentence)
    vec = [0] * len(vocab)
    for word in sentence.split():
        if word in vocab:
            vec[vocab[word]] += 1
    return torch.tensor(vec, dtype=torch.float32)

# -----------------------------
# 5) Tensors
# -----------------------------
X = torch.stack([vectorize(s, word_to_idx) for s, _ in train_data]).to(DEVICE)
y = torch.tensor([label for _, label in train_data], dtype=torch.float32).to(DEVICE)

# -----------------------------
# 6) Tiny model: linear + sigmoid (binary tone)
# -----------------------------
class TinyConcord(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = TinyConcord(input_dim=X.shape[1]).to(DEVICE)

# -----------------------------
# 7) Train (loss, optimizer, loop)
#  - Mirrors standard train/eval steps; you’d add splits for larger projects
# -----------------------------
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

# -----------------------------
# 8) Topic keys and memory
# -----------------------------
TOPIC_KEYS = [
    "ai","homework","school","music","pizza","coding","friends","exams","traffic",
    "internet","games","books","team","rules","bugs","lectures","library","math",
    "robots","stories","football","family","guitar","flowcharts","routine","nature"
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
    if assistant_text:
        memory["history"].append(("assistant", assistant_text))
    # cap history
    if len(memory["history"]) > 40:
        memory["history"] = memory["history"][-40:]

# -----------------------------
# 9) Adaptive reply engine (speaks like Copilot)
# -----------------------------
def adaptive_reply(user_text, prob, memory):
    # Confidence thresholds
    high = 0.88
    low = 0.12
    unsure_lo = 0.45
    unsure_hi = 0.55

    # Styles
    positive_styles = [
        "That’s exciting — I can feel your energy.",
        "Love that spark; you’ve got momentum.",
        "That lights me up too — keep that going.",
        "I’m hearing optimism in what you just shared."
    ]
    negative_styles = [
        "That sounds heavy — thanks for saying it.",
        "I hear the strain; that’s not easy.",
        "Feels tough — I’m with you in it.",
        "That’s a lot to carry."
    ]
    unsure_styles = [
        "Feels mixed — there’s more under the surface.",
        "A bit in-between — I’m curious about the nuance.",
        "Not fully clear yet — let’s unpack it."
    ]

    # Follow-ups
    followups_pos = [
        "What’s the best part for you?",
        "What made it click today?",
        "If you doubled down on this, what would you do next?"
    ]
    followups_neg = [
        "What’s one tiny step that could ease it?",
        "How do you usually handle moments like this?",
        "If you could change one detail, what would it be?"
    ]
    followups_unsure = [
        "What’s the detail that tips it one way?",
        "If you had to pick, which side does it lean?",
        "What’s the context around it?"
    ]

    # Hooks from topics/history (light touch)
    topics = extract_topics(user_text)
    hooks = []
    if topics:
        hooks.append(f"I’m catching {', '.join(topics)}")
    top_topics = memory_summary(memory)
    if top_topics:
        hooks.append(f"Earlier you mentioned {', '.join(top_topics)} — does this connect?")

    def hook_text():
        return (" " + random.choice(hooks) + ".") if hooks else ""

    # Branch by confidence
    if unsure_lo <= prob <= unsure_hi:
        base = random.choice(unsure_styles)
        tail = random.choice(followups_unsure)
        return f"{CONCORD_NAME}: {base}{hook_text()} {tail}"
    if prob >= high:
        base = random.choice(positive_styles)
        tail = random.choice(followups_pos)
        return f"{CONCORD_NAME}: {base}{hook_text()} {tail}"
    if prob <= low:
        base = random.choice(negative_styles)
        tail = random.choice(followups_neg)
        return f"{CONCORD_NAME}: {base}{hook_text()} {tail}"
    # Mid-range tilt
    if prob > 0.5:
        base = random.choice(positive_styles)
        tail = random.choice(followups_pos)
        return f"{CONCORD_NAME}: {base}{hook_text()} {tail}"
    else:
        base = random.choice(negative_styles)
        tail = random.choice(followups_neg)
        return f"{CONCORD_NAME}: {base}{hook_text()} {tail}"

# -----------------------------
# 10) Helpers: predict + typing
# -----------------------------
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

# -----------------------------
# 11) Commands
# -----------------------------
HELP_TEXT = (
    f"{CONCORD_NAME}: I’m built to chat naturally — concise, curious, and present.\n"
    f"{CONCORD_NAME}: Commands: /help, /quit, /summary, /topics, /reset\n"
    f"{CONCORD_NAME}: Tip: mention concrete things (coding, exams, homework, friends) for sharper replies."
)

def cmd_summary(memory):
    top = memory_summary(memory)
    if not top:
        return f"{CONCORD_NAME}: I’m still mapping your world. Say more."
    return f"{CONCORD_NAME}: What stands out so far: {', '.join(top)}."

def cmd_topics(memory):
    if not memory["topics"]:
        return f"{CONCORD_NAME}: No strong topics yet. Try referencing specifics you care about."
    items = sorted(memory["topics"].items(), key=lambda kv: kv[1], reverse=True)
    topk = [f"{k}({v})" for k, v in items[:10]]
    return f"{CONCORD_NAME}: Frequent themes: {', '.join(topk)}."

# -----------------------------
# 12) Interactive loop
# -----------------------------
def main():
    memory = {"topics": {}, "history": [], "turns": 0, "last_prob": None}
    print("\n--- Concord Integration is ready. Type your message and press Enter ---")
    typing_print("Type '/help' for tips. Type '/quit' to exit.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        cmd = user_text.lower()

        if cmd == "/quit":
            typing_print(f"{CONCORD_NAME}: Talk soon, Samridh. Keep building.")
            break
        if cmd == "/help":
            typing_print(HELP_TEXT)
            continue
        if cmd == "/reset":
            memory = {"topics": {}, "history": [], "turns": 0, "last_prob": None}
            typing_print(f"{CONCORD_NAME}: Reset complete. Fresh start.")
            continue
        if cmd == "/summary":
            typing_print(cmd_summary(memory))
            continue
        if cmd == "/topics":
            typing_print(cmd_topics(memory))
            continue

        # Predict tone, craft reply, update memory
        prob = predict_prob(user_text)
        reply = adaptive_reply(user_text, prob, memory)
        typing_print(reply)
        update_memory(memory, user_text, reply, prob)

if __name__ == "__main__":
    main()
