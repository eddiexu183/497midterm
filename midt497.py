import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Load the dataset
df = pd.read_csv('news_summary.csv', encoding='latin-1')

# Drop rows with missing values
df.dropna(subset=['ctext', 'headlines'], inplace=True)
df = df.reset_index(drop=True)

# Preprocess the text
df['ctext'] = df['ctext'].astype(str)
df['headlines'] = df['headlines'].astype(str)

# Lowercase the text
df['ctext'] = df['ctext'].apply(lambda x: x.lower())
df['headlines'] = df['headlines'].apply(lambda x: x.lower())

max_text_len = 400
max_headline_len = 60

# Filter out texts that are too long
filtered_df = df[df['ctext'].apply(lambda x: len(x.split()) <= max_text_len)]
filtered_df = filtered_df[filtered_df['headlines'].apply(lambda x: len(x.split()) <= max_headline_len)]
filtered_df = filtered_df.reset_index(drop=True)

# Tokenize the text
source_texts = filtered_df['ctext'].tolist()
target_texts = [f"sos {txt} eos" for txt in filtered_df['headlines'].tolist()]

src_tokenizer = Tokenizer()# handles the article text
src_tokenizer.fit_on_texts(source_texts)
src_sequences = src_tokenizer.texts_to_sequences(source_texts)

tgt_tokenizer = Tokenizer()# handles the headline text
tgt_tokenizer.fit_on_texts(target_texts)
tgt_sequences = tgt_tokenizer.texts_to_sequences(target_texts)

#Calculate vocabulary sizes for source and target
src_vocab_size = len(src_tokenizer.word_index) + 1
tgt_vocab_size = len(tgt_tokenizer.word_index) + 1

encoder_input_data = pad_sequences(src_sequences,  maxlen=max_text_len, padding='post')
decoder_input_data = pad_sequences(tgt_sequences,  maxlen=max_headline_len+2, padding='post')
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# 15. Define hyperparameters for the LSTM-based seq2seq model
latent_dim = 128#LSTM units
embedding_dim = 100#Embedding dimension

# Define the model
encoder_inputs = Input(shape=(max_text_len,))
# Embedding layer for the source text
enc_emb = Embedding(input_dim=src_vocab_size, 
                    output_dim=embedding_dim, 
                    trainable=True)(encoder_inputs)
# LSTM layer for the encoder
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)

encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(max_headline_len+2,))
dec_emb_layer = Embedding(input_dim=tgt_vocab_size,
                          output_dim=embedding_dim,
                          trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)
# LSTM for the decoder
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                     initial_state=encoder_states)
# Dense layer to map decoder outputs
decoder_dense = Dense(tgt_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
#The seq2seq model, trained end-to-end.
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()
decoder_target_data = np.expand_dims(decoder_target_data, -1)
#Train the model
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=5,
    validation_split=0.1
)


#encoder model
encoder_model = Model(encoder_inputs, encoder_states)
#decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#decoder embedding
dec_emb2 = dec_emb_layer(decoder_inputs)
#decoder lstm
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs
)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
#decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)
# 25. Use the same embedding layer and LSTM from training, feed in the decoder inputs plus states
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    sos_token = tgt_tokenizer.word_index['sos']
    target_seq[0, 0] = sos_token

    stop_condition = False
    decoded_sentence = ''
    # Loop to generate the headline
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in tgt_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word == '<eos>' or len(decoded_sentence.split()) >= (max_headline_len - 1):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()
