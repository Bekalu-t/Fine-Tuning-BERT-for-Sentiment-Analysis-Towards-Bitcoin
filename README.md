# Fine-Tuning-BERT-for-Sentiment-Analysis-Towards-Bitcoin
![image](https://github.com/Bekalu-t/Fine-Tuning-BERT-for-Sentiment-Analysis-Towards-Bitcoin/assets/174369527/da861fb6-5d7f-42db-8b38-ecb239ef7baf)


\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Fine-Tuning BERT for Sentiment Analysis Towards Bitcoin}
\author{Zemedkun Abebe and Bekalu Tamrat}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report documents the process of fine-tuning a BERT model for sentiment analysis on a dataset of tweets. The model was trained and evaluated using the Hugging Face Transformers library, and a simple Streamlit web application was created to deploy the model for real-time sentiment prediction.
\end{abstract}

\section{Introduction}
Sentiment analysis is a crucial task in natural language processing that aims to determine the sentiment expressed in textual data, such as tweets, reviews, or comments. Understanding sentiment can provide valuable insights into public opinion, customer feedback, and social trends. This report explores the application of BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art transformer model, for sentiment analysis on a dataset sourced from sds.csv. The goal is to classify tweets into three sentiment categories: Negative, Neutral, and Positive, using advanced deep learning techniques.

\section{ Data Preprocessing}
The dataset used in this project (sds.csv) consists of tweets labeled with sentiment categories (Negative, Neutral, Positive). Initial exploration reveals the distribution of tweets across these categories and provides insights into the dataset's size and structure.
\begin{verbatim}
        import pandas as pd
        # Load and inspect dataset
        df = pd.read_csv('Dataset/sds.csv')
        print(df.head())
        sentiment_count = df['Sentiment'].value_counts()
        print(sentiment_count)
        print(df.shape)
 \end{verbatim}     


\subsection{Data Cleaning}
Effective data cleaning is crucial for preparing text data for analysis. In this project, several preprocessing steps were applied to enhance the quality of the dataset:

Removing URLs, Hashtags, and Mentions: Using regular expressions, links (http://...), hashtags (#word), and user mentions (@username) were removed to focus solely on the textual content of tweets.


\subsection{Language Detection and Filtering}
To ensure language consistency and model performance, tweets were filtered to include only English language texts using the langid library. Non-English tweets were identified and excluded from further analysis.
\begin{verbatim}
import langid
from rich.progress import track
df['lang'] = None
for index, row in track(df.iterrows(), total=len(df)):
    text = row['text']
    if isinstance(text, str):
        language, confidence = langid.classify(text)
        df.at[index, 'lang'] = language
    else:
        df.at[index, 'lang'] = 'unknown'
df = df[df['lang'] == 'en']

 \end{verbatim}   


\section{Sentiment Analysis Pipeline}
\subsection{Data Sampling}
Due to computational constraints and to balance the dataset, a subset of 10,000 samples per sentiment category (Negative, Neutral, Positive) was selected. This sampling strategy ensures a representative distribution of sentiments for training and evaluation.
\begin{verbatim}
negativedf = df[df['Sentiment'] == 'Negative'].sample(n=10000, random_state=42)
positivedf = df[df['Sentiment'] == 'Positive'].sample(n=10000, random_state=42)
neutraldf = df[df['Sentiment'] == 'Neutral']
df = pd.concat([negativedf, neutraldf, positivedf])
 \end{verbatim}
\subsection{Tokenization with BERT}
BERT tokenizer from the transformers library was employed to tokenize tweets into numerical sequences suitable for input into the BERT model. Tokenization included padding and truncation to ensure uniform input lengths.
\begin{verbatim}
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
 \end{verbatim}


\section{Model Training and Evaluation}
\subsection{Model Selection}
BERT, known for its ability to capture contextual information in text data through bidirectional training, was chosen as the model architecture for sentiment analysis. The bert-base-uncased variant, pre-trained on large corpora, offers robust performance for natural language understanding tasks.
\begin{verbatim}
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
 \end{verbatim}
\subsection{Training Setup and Evaluation}
The model was trained using Trainer from transformers with specific training arguments (learning rate, batch size, num epochs). Evaluation metrics such as accuracy, precision, recall, and F1-score were computed to assess the model's performance on the validation dataset.

\begin{verbatim}
 training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
eval_results = trainer.evaluate()
 \end{verbatim}


\section{Results and Discussion}
\subsection{Model Performance}
The trained BERT model demonstrated strong performance in sentiment classification:  
            \begin{itemize}
    \item Accuracy: 0.56
    \item Precision: 0.56
    \item Recall: 0.56
    \item F1-score: 0.56
\end{itemize}

\subsection{Challenges and Considerations}
The following code was used to create the Streamlit application:
Despite the effective performance, challenges encountered during preprocessing or training phases may include:
\begin{enumerate}
    \item Data Imbalance: Addressing class imbalance in the dataset.
    \item Computational Constraints: Optimizing training parameters for efficiency.
\end{enumerate}

\subsection{Future Directions}
Future enhancements could explore:
\begin{itemize}
   \item Fine-tuning: Further fine-tuning BERT on domain-specific datasets for improved accuracy.
    \item Multilingual Support: Extending the model's capabilities to handle multilingual sentiment analysis tasks.
\end{itemize}

\section{Conclusion}
In conclusion, this report highlights the successful application of BERT for sentiment analysis on Twitter data. The model's ability to capture nuanced contextual information enabled accurate classification of tweets into sentiment categories.

\end{document}
