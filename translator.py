import tensorflow as tf
import pandas as pd
import numpy as np
from transformer import Transformer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Layers import create_masks
import pickle
import os

class EnglishVietnameseTranslator:
    def __init__(self, max_vocab_size=50000, max_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.eng_tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        self.vie_tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        self.model = None
        
    def load_data(self):
        # Load training data
        train_data = pd.read_csv('Data/train.csv')
        valid_data = pd.read_csv('Data/valid.csv')
        
        # Fit tokenizers on training data
        self.eng_tokenizer.fit_on_texts(train_data['en'].astype(str).values)
        self.vie_tokenizer.fit_on_texts(train_data['vi'].astype(str).values)
        
        # Add special tokens
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
        num_layers = 3
        embedding_dim = 256
        num_heads = 4
        fully_connected_dim = 1024
        input_vocab_size = len(self.eng_tokenizer.word_index) + 1
        target_vocab_size = len(self.vie_tokenizer.word_index) + 1
        max_positional_encoding_input = self.max_length
        max_positional_encoding_target = self.max_length

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
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, epochs=4, batch_size=32):
        #Load và tiền xử lý dữ liệu
        train_data, valid_data = self.load_data()
        train_eng, train_vie = self.preprocess_data(train_data)
        valid_eng, valid_vie = self.preprocess_data(valid_data)

        # # Xây dựng mô hình
        self.build_model()
        self.model.summary()

        # Vòng lặp huấn luyện
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Tạo mask cho dữ liệu huấn luyện
            enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(train_eng, train_vie[:, :-1])
            val_enc_padding_mask, val_look_ahead_mask, val_dec_padding_mask = create_masks(valid_eng, valid_vie[:, :-1])

            # Huấn luyện với đầu vào là danh sách
            history = self.model.fit(
                x=[train_eng, train_vie[:, :-1], enc_padding_mask, look_ahead_mask, dec_padding_mask],
                y=train_vie[:, 1:],
                validation_data=(
                    [valid_eng, valid_vie[:, :-1], val_enc_padding_mask, val_look_ahead_mask, val_dec_padding_mask],
                    valid_vie[:, 1:]
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
            translated.append(pr cedicted_id)
        
        # Convert indices to words
        translated_words = [self.vie_tokenizer.index_word[idx] for idx in translated]
        return ' '.join(translated_words)

    def translate_with_beam_search(self, text, beam_width=3):
        # Preprocess input text
        eng_sequence = self.eng_tokenizer.texts_to_sequences([text])[0]
        eng_padded = pad_sequences([eng_sequence], maxlen=self.max_length, padding='post', truncating='post')
        
        # Initialize beam
        start_token = self.vie_tokenizer.word_index['<start>']
        end_token = self.vie_tokenizer.word_index['<end>']
        sequences = [[list(), 0.0]]
        
        # Beam search
        for _ in range(self.max_length):
            all_candidates = list()
            for seq, score in sequences:
                if len(seq) > 0 and seq[-1] == end_token:
                    all_candidates.append((seq, score))
                    continue
                decoder_input = np.array([seq + [start_token]])
                enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(eng_padded, decoder_input)
                predictions = self.model([eng_padded, decoder_input, enc_padding_mask, look_ahead_mask, dec_padding_mask], training=False)
                predictions = predictions[:, -1, :]
                top_k = tf.math.top_k(predictions, k=beam_width)
                for i in range(beam_width):
                    candidate = [seq + [top_k.indices[0][i].numpy()], score - np.log(top_k.values[0][i].numpy())]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beam_width]
        
        # Select the best sequence
        best_sequence = sequences[0][0]
        translated_words = [self.vie_tokenizer.index_word[idx] for idx in best_sequence if idx != start_token and idx != end_token and idx in self.vie_tokenizer.index_word]
        return ' '.join(translated_words)

    def immediate_translate(self, text):
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

            # Print the current word
            print(self.vie_tokenizer.index_word.get(predicted_id, "<UNK>"), end=' ')
        
        # Convert indices to words
        return "linhlinh"
