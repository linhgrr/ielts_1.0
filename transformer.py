import tensorflow as tf
from Layers import Encoder, Decoder
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization

class Transformer(tf.keras.Model):
    def __init__(self,
                 num_layers=3,
                 embedding_dim=256,
                 num_heads=4,
                 fully_connected_dim=1024,
                 input_vocab_size=1,
                 target_vocab_size=1,
                 max_positional_encoding_input=100,
                 max_positional_encoding_target=100,
                 dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.fully_connected_dim = fully_connected_dim
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_positional_encoding_input = max_positional_encoding_input
        self.max_positional_encoding_target = max_positional_encoding_target

        self.encoder = Encoder(num_encoder_layers=num_layers,
                               d_model=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=max_positional_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_dec_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=target_vocab_size,
                               maximum_position_encoding=max_positional_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = Dense(target_vocab_size, activation='softmax')

    
    def call(self, inputs, training=None):
        """
        Forward pass for the entire Transformer
        Arguments:
            inp -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            tar -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            enc_padding_mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            final_output -- Describe me
            attention_weights - Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)

        """
        inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask = inputs

        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)

        dec_output = self.decoder(tar, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output
    
    def get_config(self):
        config = {
            'num_layers': self.num_layers,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'fully_connected_dim': self.fully_connected_dim,
            'input_vocab_size': self.input_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'max_positional_encoding_input': self.max_positional_encoding_input,
            'max_positional_encoding_target': self.max_positional_encoding_target,
        }
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        config.pop('dtype', None)
        return cls(**config)