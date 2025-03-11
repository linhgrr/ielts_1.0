# Khởi tạo translator
from translator import EnglishVietnameseTranslator

translator = EnglishVietnameseTranslator()

# Huấn luyện mô hình
# translator.train(epochs=4)

# Lưu mô hình
translator.load_model()

# Dịch một câu
text = "Hello, how are you?"
translation = translator.translate_with_beam_search(text)
print(f"Input: {text}")
print(f"Translation: {translation}")