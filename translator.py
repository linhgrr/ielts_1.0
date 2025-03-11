import tensorflow as tf
import pandas as pd
import numpy as np
from transformer import Transformer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Layers.mask import create_masks
import pickle
import os
import re
import string
import matplotlib.pyplot as plt
from keras.saving import register_keras_serializable
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = float(d_model)  # Ensure d_model is a float to match the config
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # Common learning rate schedule for transformers
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        # Must return the same config keys as in the error message
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
    
    

class EnglishVietnameseTranslator:
    def __init__(self, max_vocab_size=50000, max_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.eng_tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        self.vie_tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        self.model = None

    def normalize_text(text):
        text = text.lower()  # Chuyển thành chữ thường
        text = text.translate(str.maketrans('', '', string.punctuation))  # Xóa dấu câu
        text = re.sub(r'[^a-zA-Z0-9\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]', '', text)  # Giữ chữ cái và số
        text = ' '.join(text.split())  # Xóa khoảng trắng thừa
        return text
    
    def filter_text_length(data):
        min_length = 2  # Tối thiểu 2 từ
        max_length = 50  # Tối đa 50 từ
        data = data[data['en'].apply(lambda x: min_length <= len(x.split()) <= max_length)]
        data = data[data['vi'].apply(lambda x: min_length <= len(x.split()) <= max_length)]

        return data

        
    def load_data(self):
        train_data = pd.read_csv('Data/train.csv')
        valid_data = pd.read_csv('Data/valid.csv')

        # Loại bỏ giá trị NaN
        train_data.dropna(inplace=True)
        valid_data.dropna(inplace=True)

        train_data = self.filter_text_length(train_data)
        valid_data = self.filter_text_length(valid_data)
        # Áp dụng chuẩn hóa văn bản
        train_data['en'] = train_data['en'].apply(self.normalize_text)
        train_data['vi'] = train_data['vi'].apply(self.normalize_text)
        valid_data['en'] = valid_data['en'].apply(self.normalize_text)
        valid_data['vi'] = valid_data['vi'].apply(self.normalize_text)

        # Fit tokenizer trên dữ liệu đã chuẩn hóa
        self.eng_tokenizer.fit_on_texts(train_data['en'].values)
        self.vie_tokenizer.fit_on_texts(train_data['vi'].values)

        # Thêm token đặc biệt cho tiếng Việt
        self.vie_tokenizer.word_index['<start>'] = len(self.vie_tokenizer.word_index) + 1
        self.vie_tokenizer.word_index['<end>'] = len(self.vie_tokenizer.word_index) + 1
        self.vie_tokenizer.index_word[self.vie_tokenizer.word_index['<start>']] = '<start>'
        self.vie_tokenizer.index_word[self.vie_tokenizer.word_index['<end>']] = '<end>'

        return train_data, valid_data
    
    def preprocess_data(self, data):
        # Convert text to sequences
        eng_sequences = self.eng_tokenizer.texts_to_sequences(data['en'].astype(str).values)
        vie_sequences = self.vie_tokenizer.texts_to_sequences(data['vi'].astype(str).values)
        
        # Add start and end tokens to Vietnamese sequences
        vie_sequences = [[self.vie_tokenizer.word_index['<start>']] + seq + [self.vie_tokenizer.word_index['<end>']] 
                        for seq in vie_sequences]
        
        # Pad sequences
        eng_padded = pad_sequences(eng_sequences, maxlen=self.max_length, padding='post', truncating='post')
        vie_padded = pad_sequences(vie_sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        return eng_padded, vie_padded
    
    def build_model(self):
        num_layers = 4
        embedding_dim = 256
        num_heads = 4
        fully_connected_dim = 1024
        input_vocab_size = len(self.eng_tokenizer.word_index) + 1
        target_vocab_size = len(self.vie_tokenizer.word_index) + 1
        max_positional_encoding_input = self.max_length
        max_positional_encoding_target = self.max_length

        learning_rate = CustomSchedule(embedding_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.model = Transformer(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            fully_connected_dim=fully_connected_dim,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            max_positional_encoding_input=max_positional_encoding_input,
            max_positional_encoding_target=max_positional_encoding_target
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )

    def train(self, epochs=4, batch_size=32):
        #Load và tiền xử lý dữ liệu
        train_data, valid_data = self.load_data()
        train_eng, train_vie = self.preprocess_data(train_data)
        valid_eng, valid_vie = self.preprocess_data(valid_data)

        # # Xây dựng mô hình
        self.build_model()

        # Vòng lặp huấn luyện
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(train_eng, train_vie[:, :-1])
            val_enc_padding_mask, val_look_ahead_mask, val_dec_padding_mask = create_masks(valid_eng, valid_vie[:, :-1])

            # Tạo sample_weight để bỏ qua padding
            sample_weight = np.ones_like(train_vie[:, 1:])
            sample_weight[train_vie[:, 1:] == 0] = 0  # Trọng số 0 cho padding
            val_sample_weight = np.ones_like(valid_vie[:, 1:])
            val_sample_weight[valid_vie[:, 1:] == 0] = 0

            history = model.fit(
                x=[train_eng, train_vie[:, :-1], enc_padding_mask, look_ahead_mask, dec_padding_mask],
                y=train_vie[:, 1:],
                sample_weight=sample_weight,
                validation_data=(
                    [valid_eng, valid_vie[:, :-1], val_enc_padding_mask, val_look_ahead_mask, val_dec_padding_mask],
                    valid_vie[:, 1:],
                    val_sample_weight
                ),
                batch_size=batch_size,
                epochs=1
            )
        
    
    def save_model(self, path='translation_model'):
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save tokenizers
        with open(f'{path}/eng_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.eng_tokenizer, handle)
        with open(f'{path}/vie_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.vie_tokenizer, handle)
        
        # Save model
        self.model.save(f'{path}/model.keras')
    
    def load_model(self, path='translation_model'):
        # Load tokenizers
        with open(f'{path}/eng_tokenizer.pickle', 'rb') as handle:
            self.eng_tokenizer = pickle.load(handle)
        with open(f'{path}/vie_tokenizer.pickle', 'rb') as handle:
            self.vie_tokenizer = pickle.load(handle)
        
        # Load model
        self.model = tf.keras.models.load_model(
            f'{path}/model.keras',
            custom_objects={'Transformer': Transformer}
        )
    
    def translate(self, text):
        # Preprocess input text
        eng_sequence = self.eng_tokenizer.texts_to_sequences([text])[0]
        eng_padded = pad_sequences([eng_sequence], maxlen=self.max_length, padding='post', truncating='post')
        
        # Initialize decoder input with start token
        decoder_input = np.array([[self.vie_tokenizer.word_index['<start>']]])
        
        # Generate translation
        translated = []
        for i in range(self.max_length):
            # Create masks
            enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(eng_padded, decoder_input)
            
            # Get prediction
            predictions = self.model([eng_padded, decoder_input, enc_padding_mask, look_ahead_mask, dec_padding_mask], 
                        training=False)
            
            # Get the last word prediction
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            # Convert predicted_id to a numpy array and then to a Python integer
            predicted_id = predicted_id.numpy()[0][0]

            # Stop if end token is predicted
            if predicted_id == self.vie_tokenizer.word_index['<end>']:
                break
                
            # Add predicted word to decoder input
            decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)
            translated.append(predicted_id)
        
        # Convert indices to words
        translated_words = [self.vie_tokenizer.index_word[idx] for idx in translated]
        return ' '.join(translated_words)


    def translate_beam_search(self,text, beam_width=5):
        # Chuẩn hóa văn bản đầu vào
        text = self.normalize_text(text)
        eng_sequence = self.eng_tokenizer.texts_to_sequences([text])
        eng_sequence = pad_sequences(eng_sequence, maxlen=self.max_length, padding='post')

        start_token = self.vie_tokenizer.word_index['<start>']
        end_token = self.vie_tokenizer.word_index['<end>']

        # Khởi tạo với chuỗi bắt đầu chỉ chứa <start>
        sequences = [(np.array([[start_token]]), 0.0)]

        for _ in range(self.max_length):
            all_candidates = []
            for seq, score in sequences:
                # Nếu chuỗi đã kết thúc, không mở rộng thêm
                if seq[0, -1] == end_token:
                    all_candidates.append((seq, score))
                    continue

                enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(eng_sequence, seq)
                predictions = self.model.predict([eng_sequence, seq, enc_padding_mask, look_ahead_mask, dec_padding_mask])
                # Lấy dự đoán cho token cuối cùng (dạng xác suất)
                preds = predictions[0, -1, :]  # shape: (vocab_size,)

                # Lấy chỉ số của top beam_width từ các xác suất cao nhất
                top_indices = np.argsort(preds)[-beam_width:][::-1]

                for word_id in top_indices:
                    prob = preds[word_id]
                    # Bỏ qua nếu xác suất bằng 0 để tránh lỗi log(0)
                    if prob == 0:
                        continue
                    new_score = score + np.log(prob)
                    # Nếu là token kết thúc, không mở rộng thêm
                    if word_id == end_token:
                        candidate_seq = seq
                    else:
                        candidate_seq = np.concatenate([seq, np.array([[word_id]])], axis=1)
                    all_candidates.append((candidate_seq, new_score))

            # Sắp xếp các candidate theo điểm số và chọn beam_width chuỗi có điểm cao nhất
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Nếu tất cả các chuỗi đều kết thúc, dừng vòng lặp
            if all(seq[0, -1] == end_token for seq, _ in sequences):
                break

        # Lấy chuỗi có điểm cao nhất
        best_seq = sequences[0][0][0]  # Chuyển sang dạng 1D
        decoded_sentence = ' '.join([self.vie_tokenizer.index_word[word_id] for word_id in best_seq if word_id not in [start_token, end_token]])

        return decoded_sentence.strip()
