# Concord Integration — interactive conversational micro-model
# Author: Samridh
# Goal: Listen, remember, and answer naturally with confidence-aware tone and playful interactivity

import re
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 0) Persona and training settings
# -----------------------------
CONCORD_NAME = "Concord Integration"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True
EPOCHS = 50000        # push higher for near-0 loss polish
LR = 0.001
LOG_INTERVAL = 10000
TYPE_SPEED = 0.01       # 0 = instant; set e.g., 0.01 for "typing" effect

if PRINT_DEVICE:
    print(f"{CONCORD_NAME} starting on device: {DEVICE}")

# -----------------------------
# 1) Text normalization
# -----------------------------
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------------
# 2) Expanded training data (balanced, 100+ examples)
# -----------------------------
train_data = [
    # Positive — emotions, school, tech, hobbies, daily life
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

    # Negative — emotions, school stress, tech frustrations, daily pains
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
# 3) Build vocabulary dynamically from normalized training data
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
# 4) Vectorizer (bag-of-words on normalized text)
# -----------------------------
def vectorize(sentence, vocab):
    sentence = normalize(sentence)
    vec = [0] * len(vocab)
    for word in sentence.split():
        if word in vocab:
            vec[vocab[word]] += 1
        # unseen words ignored in basic BoW
    return torch.tensor(vec, dtype=torch.float32)

# -----------------------------
# 5) Create tensors
# -----------------------------
X = torch.stack([vectorize(s, word_to_idx) for s, _ in train_data]).to(DEVICE)
y = torch.tensor([label for _, label in train_data], dtype=torch.float32).to(DEVICE)

# -----------------------------
# 6) Model: simple linear + sigmoid
# -----------------------------
class TinyConcord(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # probability of "positive tone"

model = TinyConcord(input_dim=X.shape[1]).to(DEVICE)

# -----------------------------
# 7) Loss and optimizer
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# -----------------------------
# 8) Training loop
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % LOG_INTERVAL == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:6d} | Loss: {loss.item():.6f}")

# -----------------------------
# 9) Memory and topic tracking
# -----------------------------
TOPIC_KEYS = [
    "ai", "homework", "school", "music", "pizza", "coding", "friends",
    "exams", "traffic", "internet", "games", "books", "team", "rules",
    "bugs", "lectures", "library", "math", "robots", "stories", "football",
    "family", "guitar", "flowcharts", "routine", "nature"
]

def extract_topics(text):
    t = normalize(text).split()
    found = [k for k in TOPIC_KEYS if k in t]
    return found

def update_memory(memory, topics, sentiment_prob):
    for t in topics:
        memory["topics"][t] = memory["topics"].get(t, 0) + 1
    memory["last_prob"] = sentiment_prob
    memory["turns"] += 1

def memory_summary(memory):
    if not memory["topics"]:
        return None
    top = sorted(memory["topics"].items(), key=lambda kv: kv[1], reverse=True)
    top_topics = [k for k, v in top[:3]]
    return top_topics

# -----------------------------
# 10) Interactive reply engine
# -----------------------------
def typing_print(text):
    if TYPE_SPEED <= 0:
        print(text)
        return
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(TYPE_SPEED)
    print()

def build_reply(prob, sentence, memory, topics):
    high = 0.88
    low = 0.12
    unsure_lo = 0.45
    unsure_hi = 0.55

    style_positive = [
        "Love that energy.",
        "That spark is contagious.",
        "I can feel your momentum.",
        "That’s a win-worthy vibe."
    ]
    style_negative = [
        "That sounds rough.",
        "I hear the strain in that.",
        "Feels heavy—thanks for saying it.",
        "That’s not easy to sit with."
    ]
    style_unsure = [
        "Mixed signals here.",
        "Feels a bit in-between.",
        "Not fully clear yet."
    ]

    # Topic-aware hooks
    hooks = []
    if topics:
        hooks.append(f"I’m catching themes like {', '.join(topics)}.")
    top_topics = memory_summary(memory)
    if top_topics:
        hooks.append(f"Earlier you mentioned {', '.join(top_topics)}—does this connect?")

    # Confidence-based response with variety
    if unsure_lo <= prob <= unsure_hi:
        base = random.choice(style_unsure)
        followups = [
            "What’s the twist behind it?",
            "Tell me the detail you’re holding back.",
            "If you had to choose, which side does it lean?"
        ]
        return f"{CONCORD_NAME}: {base} {random.choice(hooks) if hooks else ''} {random.choice(followups)}"

    if prob >= high:
        base = random.choice(style_positive)
        followups = [
            "What’s the best part about it for you?",
            "How would you double down on this feeling?",
            "What made it click today?"
        ]
        return f"{CONCORD_NAME}: {base} {random.choice(hooks) if hooks else ''} {random.choice(followups)}"

    if prob <= low:
        base = random.choice(style_negative)
        followups = [
            "What’s the smallest change that could ease it?",
            "How do you usually handle this?",
            "Want to unpack one concrete trigger?"
        ]
        return f"{CONCORD_NAME}: {base} {random.choice(hooks) if hooks else ''} {random.choice(followups)}"

    # Mid-range
    if prob > 0.5:
        base = random.choice(style_positive)
        followups = [
            "What’s one thing you’d keep exactly as it is?",
            "If this were a habit, how would you keep it going?",
            "What does this say about what you value?"
        ]
        return f"{CONCORD_NAME}: {base} {random.choice(hooks) if hooks else ''} {random.choice(followups)}"
    else:
        base = random.choice(style_negative)
        followups = [
            "If you could change one detail, what would it be?",
            "Where does this feeling usually show up?",
            "What’s a tiny step you’d try first?"
        ]
        return f"{CONCORD_NAME}: {base} {random.choice(hooks) if hooks else ''} {random.choice(followups)}"

# -----------------------------
# 11) Prediction helper
# -----------------------------
def predict_sentence(sentence):
    with torch.no_grad():
        x = vectorize(sentence, word_to_idx).to(DEVICE).unsqueeze(0)
        p = model(x).item()
        return p

# -----------------------------
# 12) Commands and mini-interactions
# -----------------------------
HELP_TEXT = (
    f"{CONCORD_NAME}: I understand everyday phrases (emotions, school, coding, hobbies, daily life).\n"
    f"{CONCORD_NAME}: Commands: /help, /quit, /summary, /challenge, /topics\n"
    f"{CONCORD_NAME}: Tip: add concrete words (e.g., exams, homework, coding, friends) for sharper replies."
)

def challenge():
    prompts = [
        "Name one habit that makes your day 1% better.",
        "Describe a tiny win from this week.",
        "Share a frustration and one small fix you’d try."
    ]
    return f"{CONCORD_NAME}: {random.choice(prompts)}"

def summary(memory):
    top = memory_summary(memory)
    if not top:
        return f"{CONCORD_NAME}: I’m still getting a feel for your themes. Say more."
    return f"{CONCORD_NAME}: Here’s what stands out so far: {', '.join(top)}."

def topics_list(memory):
    if not memory["topics"]:
        return f"{CONCORD_NAME}: No strong topics yet. Try mentioning specific things you care about."
    items = sorted(memory["topics"].items(), key=lambda kv: kv[1], reverse=True)
    topk = [f"{k}({v})" for k, v in items[:10]]
    return f"{CONCORD_NAME}: Your frequent themes: {', '.join(topk)}."

# -----------------------------
# 13) Interactive loop (listen, remember, answer)
# -----------------------------
memory = {"topics": {}, "last_prob": None, "turns": 0}

print("\n--- Concord Integration is ready. Type your sentence and press Enter ---")
typing_print("Type '/help' for tips. Type '/quit' to exit.\n")

while True:
    user_inp = input("You: ").strip()
    if not user_inp:
        continue
    cmd = user_inp.lower()

    if cmd == "/quit":
        typing_print(f"{CONCORD_NAME}: Talk soon, Samridh. Keep building.")
        break
    if cmd == "/help":
        typing_print(HELP_TEXT)
        continue
    if cmd == "/challenge":
        typing_print(challenge())
        continue
    if cmd == "/summary":
        typing_print(summary(memory))
        continue
    if cmd == "/topics":
        typing_print(topics_list(memory))
        continue

    # Predict + remember + reply
    prob = predict_sentence(user_inp)
    t = extract_topics(user_inp)
    update_memory(memory, t, prob)

    reply = build_reply(prob, user_inp, memory, t)
    typing_print(reply)
