import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import plotly.graph_objects as go

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
st.title("₿ Sentiment Analysis Towards BitCoin with Fine-Tuned BERT ₿")
st.write("Enter text to analyze sentiment:")

user_input = st.text_area("Text Input", "Type your text here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        
        # Define the gauge chart with simulated gradient
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': f"Sentiment: {sentiment}"},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 0.1], 'color': "rgba(255,1,1,1)"},
                    {'range': [0.1, 0.2], 'color': "rgba(255,84,0,1)"},
                    {'range': [0.2, 0.3], 'color': "rgba(255,167,0,1)"},
                    {'range': [0.3, 0.4], 'color': "rgba(255,214,0,1)"},
                    {'range': [0.4, 0.5], 'color': "rgba(255,214,0,1)"},
                    {'range': [0.5, 0.6], 'color': "rgba(241,255,1,1)"},
                    {'range': [0.6, 0.7], 'color': "rgba(198,255,0,1)"},
                    {'range': [0.7, 0.8], 'color': "rgba(155,255,0,1)"},
                    {'range': [0.8, 0.9], 'color': "rgba(9,255,0,1)"},
                    {'range': [0.9, 1], 'color': "rgba(9,255,0,1)"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence}}))

        # Display the gauge chart
        st.plotly_chart(fig)
    else:
        st.write("Please enter some text to analyze.")

# Save model button
if st.button("Save Model"):
    model.save_pretrained('./fine-tuned-bert')
    tokenizer.save_pretrained('./fine-tuned-bert')
    st.write("Model and tokenizer saved successfully.")
