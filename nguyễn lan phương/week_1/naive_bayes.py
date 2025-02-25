import math
from collections import defaultdict

# Tập dữ liệu huấn luyện
train_data = [
    ("I love this movie", "Positive"),
    ("I hate this movie", "Negative"),
    ("This movie is amazing", "Positive"),
    ("I dislike this film", "Negative"),
    ("Fantastic movie", "Positive"),
    ("Terrible film", "Negative")
]

# Tiền xử lý: Tách từ
def tokenize(sentence):
    return sentence.lower().split()

# Đếm số lượng từ trong từng nhãn
word_counts = {"Positive": defaultdict(int), "Negative": defaultdict(int)}
class_counts = {"Positive": 0, "Negative": 0}
vocab = set()

# Duyệt qua dữ liệu huấn luyện
for sentence, label in train_data:
    words = tokenize(sentence)
    class_counts[label] += 1  # Đếm số câu thuộc từng nhãn
    for word in words:
        word_counts[label][word] += 1  # Đếm số lần xuất hiện của từ trong nhãn
        vocab.add(word)  # Thêm từ vào từ vựng

# Tính P(C) - Xác suất tiên nghiệm của từng nhãn
total_docs = sum(class_counts.values())
priors = {label: class_counts[label] / total_docs for label in class_counts}

# Tính P(w|C) với Laplacian Smoothing (α = 1)
alpha = 1
vocab_size = len(vocab)
word_probs = {}

for label in ["Positive", "Negative"]:
    total_words = sum(word_counts[label].values())  # Tổng số từ trong nhãn
    word_probs[label] = {
        word: (word_counts[label][word] + alpha) / (total_words + alpha * vocab_size)
        for word in vocab
    }

# Hàm tính Log-Likelihood
def compute_log_likelihood(sentence):
    words = tokenize(sentence)
    log_likelihoods = {}

    for label in ["Positive", "Negative"]:
        log_likelihoods[label] = math.log(priors[label])  # log P(C)
        for word in words:
            if word in word_probs[label]:  # Chỉ tính những từ có trong từ vựng
                log_likelihoods[label] += math.log(word_probs[label][word])

    return log_likelihoods

# Hàm phân loại cảm xúc
def classify(sentence):
    log_likelihoods = compute_log_likelihood(sentence)
    return max(log_likelihoods, key=log_likelihoods.get)  # Chọn nhãn có log-likelihood cao nhất

# **📌 TEST: Dự đoán cảm xúc**
test_sentences = [
    "I love this film",
    "This movie is terrible",
    "Fantastic acting",
    "I dislike this movie"
]

for sentence in test_sentences:
    print(f"'{sentence}' ➝ {classify(sentence)}")
