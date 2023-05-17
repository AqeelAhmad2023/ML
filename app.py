import re
import string
import pandas as pd
import pickle
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences


def textpro(text):
    text=text.lower()
    text= re.sub('\[.*?\]', ' ', text)
    text= re.sub('\\W', ' ', text)
    text= re.sub('https?://\S+|www\.\S+', ' ',text)
    text= re.sub('<.*?>+', ' ',text)
    text= re.sub('[%s]' % re.escape(string.punctuation), ' ',text)
    text= re.sub('\n', ' ',text)
    text= re.sub('\w*\d\w*', ' ',text)
    return text


pickled_model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vector.pkl', 'rb'))


# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
# Load the saved model
loaded_model = load_model("keras_lstm_fakenews_detector.h5")






# website
st.title('Fake News Detector Website')

algorithms = {
    "Logistic Regression": 1,
    "RNN": 2
}

# Create a dropdown list to select the algorithm
selected_algorithm = st.selectbox("##### Select an algorithm:", list(algorithms.keys()))

# Get the selected algorithm object from the dictionary
selected_model = algorithms[selected_algorithm]

# Use the selected algorithm object for prediction

st.write("\t")
st.write("\t")
st.write("\t")
st.write("\t")

# Use st.beta_columns() to create two columns
col1, col2 = st.columns([7, 1])
# Get user input in the first column
with col1:
    input_text = st.text_input("##### Enter news Article:")

# Create a button to display the input text in the second column
with col2:
    st.write("")
    st.write("")
    st.write("")

    a=st.button("Predict")
st.write("#### News Article:")
st.write(input_text)


def predict_news(raw_text, model, tokenizer, maxlen=128):
    # Preprocess the input text
    input_seq = tokenizer.texts_to_sequences([raw_text])
    input_padded = pad_sequences(input_seq, maxlen=maxlen)

    # Make the prediction
    probabilities = model.predict(input_padded)
    prediction = probabilities.argmax(axis=-1)

    return prediction


# Load the saved tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

#load the classification report
with open("class_R.pkl", "rb") as f:
    report_rnn = pickle.load(f)

#load the classification report
with open("acu_lgr.pkl", "rb") as f:
    report_lgr = pickle.load(f)





def prediction(news):
    testing_news = {"text": [news]}
    new_df_test = pd.DataFrame(testing_news)
    new_df_test["text"] = new_df_test["text"].apply(textpro)
    new_x_test = new_df_test["text"]
    raw_text = input_text
    new_xvector_test = vectorizer.transform(new_x_test)
    if selected_model == 1:
        prediction = pickled_model.predict(new_xvector_test)
    elif selected_model ==2:
        prediction = predict_news(raw_text, loaded_model, tokenizer)
    return prediction[0]


st.write("#### prediction:")

if input_text or a == True:
    pred = prediction(input_text)
    if pred == 1 or prediction==1:
        st.write('The News is Real')
    else:
        st.write('The News may be Fake')

if selected_model ==2:
    st.write("#### Accuracy:\n")
    st.write(report_rnn)
else:
    st.write("#### Accuracy:\n")
    st.write(report_lgr)


st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("Created by Aqeel & Azra")
