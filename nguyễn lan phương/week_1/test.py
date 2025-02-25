import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Tải dữ liệu stopwords từ NLTK
nltk.download('stopwords')

# Khởi tạo từ điển tần suất
pos_freqs = {}
neg_freqs = {}

# Hàm xây dựng tần suất từ
def build_freqs(tweets, labels):
    for tweet, label in zip(tweets, labels):
        processed_tweet = process_tweet(tweet)  # Xử lý tweet trước khi đếm
        words = processed_tweet.split()
        freq_dict = pos_freqs if label == 1 else neg_freqs
        for word in words:
            freq_dict[word] = freq_dict.get(word, 0) + 1

# Hàm tiền xử lý tweet
def process_tweet(tweet):
    # Chuyển đổi thành chữ thường
    tweet = tweet.lower()
    # Loại bỏ các handle (@username)
    tweet = re.sub(r'@\w+', '', tweet)
    # Loại bỏ các URL
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'www\S+', '', tweet)
    # Tách từ
    words = tweet.split()
    # Loại bỏ stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    # Ghép lại thành câu
    tweet = ' '.join(words)
    return tweet

# Hàm trích xuất đặc trưng
def extract_features(tweet):
    word_l = process_tweet(tweet).split()
    x = np.zeros((1, 3))
    x[0, 0] = 1  # Bias term
    x[0, 1] = sum(pos_freqs.get(word, 0) for word in word_l)  # Tổng tần suất từ tích cực
    x[0, 2] = sum(neg_freqs.get(word, 0) for word in word_l)  # Tổng tần suất từ tiêu cực
    return x.flatten()

# Tạo dữ liệu mẫu lớn hơn
tweets = [
    'I am happy because I am learning',
    'I am happy',
    'I am sad, I am not learning',
    'Learning is fun and exciting',
    'I am not happy with the results',
    'The weather is gloomy and I feel sad',
    'I am thrilled with the new project',
    'I am disappointed with the delay',
    'I am feeling great today',
    'I am not satisfied with the service',
    'This is the best day of my life',
    'I am so excited about the upcoming event',
    'I feel terrible about what happened',
    'The movie was fantastic and I enjoyed it a lot',
    'I am frustrated with the slow progress',
    'I am overjoyed with the results',
    'I am bored and have nothing to do',
    'The food was delicious and the service was excellent',
    'I am angry about the poor customer service',
    'I am grateful for all the support I received',
    'I am worried about the upcoming exam',
    'I am delighted with the outcome',
    'I am annoyed by the constant noise',
    'I am content with my life right now',
    'I am shocked by the sudden news',
    'I am enthusiastic about the new opportunities',
    'I am depressed because of the recent events',
    'I am optimistic about the future',
    'I am stressed out with all the work',
    'I am amazed by the incredible performance'
]

labels = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Xây dựng từ điển tần suất
build_freqs(tweets, labels)

# Trích xuất đặc trưng cho tất cả tweet
X = np.array([extract_features(tweet) for tweet in tweets])

# Chia tập dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Khởi tạo và huấn luyện mô hình SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Dự đoán:", y_pred)
print("Nhãn thực tế:", y_test)
print("Độ chính xác:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))