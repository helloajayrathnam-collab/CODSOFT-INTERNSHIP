import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
import os

# ---------------------------
# Feature Extraction using ResNet50 (Pre-trained)
# ---------------------------

def extract_image_features(img_path):
    base_model = ResNet50(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    features = model.predict(x, verbose=0)
    return features

# ---------------------------
# Dummy Tokenizer and Vocabulary
# ---------------------------

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(["startseq", "a", "man", "riding", "horse", "on", "beach", "endseq"])
vocab_size = len(tokenizer.word_index) + 1
max_len = 7

# ---------------------------
# Caption Generator Model (CNN encoder + LSTM decoder)
# ---------------------------

def build_model(vocab_size, max_len):
    input1 = Input(shape=(2048,))
    fe1 = Dense(256, activation='relu')(input1)

    input2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(input2)
    se2 = LSTM(256)(se1)

    decoder = Add()([fe1, se2])
    decoder = Dense(256, activation='relu')(decoder)
    output = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[input1, input2], outputs=output)
    return model

model = build_model(vocab_size, max_len)

# ---------------------------
# Caption Generation (Greedy Search)
# ---------------------------

def generate_caption(feature, tokenizer, max_len):
    in_text = 'startseq'
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)

        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)

        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return caption

# ---------------------------
# Main Function
# ---------------------------

def main():
    image_path = "sample.jpg"  # ⚠️ Replace with your image file
    if not os.path.exists(image_path):
        print("Image not found! Place an image named 'sample.jpg' in the folder.")
        return

    features = extract_image_features(image_path)
    caption = generate_caption(features, tokenizer, max_len)

    print("Generated Caption:", caption)

    img = image.load_img(image_path)
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
