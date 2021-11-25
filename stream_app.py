import numpy as np
import streamlit as st
import pickle
import time
import streamlit as st
from tensorflow.keras.models import  Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,LSTM ,Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.text import Tokenizer

st.markdown(
    """
    <style>
    body {
  background-image: url('https://wallpapercave.com/wp/fLmAxoy.jpg');
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-size: cover;
}
</style>
 """,
    unsafe_allow_html=True
)




st.cache(allow_output_mutation=True)
def main():
    st.markdown("<h1 style ='text-align: centre;color:red;'>Sentence Generation Using LSTM</h1>",unsafe_allow_html=True)
    st.markdown("<h3 style ='text-align: centre;color:red;'>Enetr the number of words to be genreated</h3>",unsafe_allow_html=True)

    text = []
    with open('s1_2.txt', encoding='unicode_escape') as f:
        for i in f:
            text.append(i)

    final_doc =' '.join(text)

    @st.cache(allow_output_mutation=True)
    def clean_text(doc):
        import string
        tokens = doc.split(" ")  # white space sep
        punc = str.maketrans("", "", string.punctuation)  # all punc
        tokens = [w.translate(punc) for w in tokens]  # remove punc
        tokens = [word for word in tokens if word.isalpha()]  # only alpha
        tokens = [word.lower() for word in tokens]  # lower
        return tokens

    tokens = clean_text(final_doc)
    input_length = 30 + 1
    lines = []

    for i in range(input_length, len(tokens)):
        seq = tokens[i - input_length: i]  # 0 to inp length
        line = " ".join(seq)  # join to make inp sequence
        lines.append(line)  # append in list

    ## Tokenize

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)  # tokens applied on lines
    seq = tokenizer.texts_to_sequences(lines)
    seq = np.array(seq)
    voc_size = len(tokenizer.word_index) + 1  # as it was started from 0
    seq_length = 30

    ##Model

    model = Sequential()
    model.add(Embedding(voc_size, 30, input_length=seq.shape[1]))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(voc_size / 2, activation='relu'))
    model.add(Dense(voc_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    path_checkpoint = 'mymodel1_2.h5'
    from tensorflow.keras.models import load_model
    model = load_model('mymodel.h5', compile=False)

    ## generate func
    def generate_text(model, tokenizer, text_seq_length, seed_text, n_words):
        final_ans = []
        for n in range(n_words):
            encoded = tokenizer.texts_to_sequences([seed_text])[0]
            encoded = pad_sequences([encoded], maxlen=text_seq_length, truncating='pre')
            y_pred = np.argmax(model.predict(encoded), axis=-1)
            pred_word = " "
            for word, index in tokenizer.word_index.items():
                if index == y_pred:
                    pred_word = word
                    break
            seed_text = seed_text + " " + pred_word
            final_ans.append(pred_word)
        return " ".join(final_ans)

    num_of_word = st.selectbox('',[10,20,30,40,50,60,70])
    st.markdown("<h3 style='text-align: center; color:Red;'> Enter the index:</h3>",unsafe_allow_html=True)
    ind=st.slider('',0,len(lines))
    if ind!=0:
        inp =lines[ind]
        st.markdown("<h4 style='text-align: center; color: White;background :rgba(66, 240, 50, 0.6);'>Input:</h2>",unsafe_allow_html=True)
        st.markdown(f""" <h3 style='text-align: center; color: White;background :rgba(53, 184, 240, 0.9);'>{inp}</h3>""",unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: White;background :rgba(66, 240, 50, 0.6)'>Predicted Output:</h2>", unsafe_allow_html=True)
        st.markdown(f"""<h3 style='text-align: center; color: white;background :rgba(53, 184, 240, 0.9);'> {generate_text(model, tokenizer, seq_length, inp, num_of_word)}</h3>""",unsafe_allow_html=True)

if __name__ == "__main__":
    main()
