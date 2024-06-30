import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('./fine-tuned-bert')
tokenizer = BertTokenizer.from_pretrained('./fine-tuned-bert')

# Define sentiment labels
sentiment_labels = ['Negative', 'Neutral', 'Positive']

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = np.argmax(probabilities.numpy(), axis=1)[0]
    confidence = probabilities[0][predicted_class].item()
    return sentiment_labels[predicted_class], confidence

# Streamlit UI
st.title("Sentiment Analysis with Fine-Tuned BERT")
st.write("Enter text to analyze sentiment:")

user_input = st.text_area("Text Input", "Type your text here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.write("Please enter some text to analyze.")

# Save model button
if st.button("Save Model"):
    model.save_pretrained('./fine-tuned-bert')
    tokenizer.save_pretrained('./fine-tuned-bert')
    st.write("Model and tokenizer saved successfully.")
