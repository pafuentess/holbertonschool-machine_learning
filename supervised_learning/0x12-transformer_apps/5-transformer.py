#!/usr/bin/env python3
""" doc """

import tensorflow.compat.v2 as tf


def positional_encoding(max_seq_len, dm):
    """ doc """
    encode = np.zeros([max_seq_len, dm])

    for i in range(dm):
        for p in range(max_seq_len):
            encode[p, i] = p / np.power(10000, (2 * (i // 2) / dm))

    encode[:, 0::2] = np.sin(encode[:, 0::2])
    encode[:, 1::2] = np.cos(encode[:, 1::2])

    return encode


def sdp_attention(Q, K, V, mask=None):
    """ doc """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, dm, h):
        """ doc """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """ doc """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """ doc """
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        output = self.linear(concat_attention)
        return output, attention_weights


class EncoderBlock(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ doc """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """ doc """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ doc """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ doc """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, encoder_output,
                                               encoder_output,
                                               padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Encoder(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """ doc """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """ doc """
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """ doc """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ doc """
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        return x


class Transformer(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """ doc """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        return self.linear(dec_output)
