📰 Fake News Detection using Machine Learning

📌 Overview
This project is focused on detecting fake news articles using Natural Language Processing (NLP) and Machine Learning algorithms. Given a news article or headline, the system predicts whether it is real or fake.

🚀 Features
Preprocessing and cleaning of news text
Exploratory Data Analysis (EDA)
TF-IDF vectorization of text
Training multiple ML models (Logistic Regression, Naive Bayes, etc.)
Evaluation using metrics like accuracy, precision, recall, F1-score
User-friendly predictions on new data

🛠️ Tech Stack
Programming Language: Python
Libraries & Tools:
Pandas, NumPy
Scikit-learn
NLTK / spaCy (for NLP)
Matplotlib / Seaborn (for EDA)
Jupyter Notebook

📂 Dataset
Dataset: Fake and Real News Dataset
Columns include: title, text, subject, date, label

📈 Model Workflow
Data Cleaning: Removing punctuation, lowercasing, removing stopwords, stemming/lemmatization
Text Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency)
Train-Test Split
Model Training: Logistic Regression / Naive Bayes
Model Evaluation: Confusion matrix, Accuracy, Precision, Recall, F1-score

✅ Results
Accuracy: ~93% with Logistic Regression (varies based on preprocessing)
Precision/Recall balanced for both fake and real classes
