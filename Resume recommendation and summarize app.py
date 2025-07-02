import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pdfplumber
import docx
from transformers import pipeline

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model, vectorizer, and label encoder
@st.cache_resource
def load_artifacts():
    with open('resume_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_artifacts()

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def preprocess_text(text):
    # Check if text is string
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove very short words (length <= 2)
    tokens = [word for word in tokens if len(word) > 2]
    # Join back to string
    processed_text = ' '.join(tokens)
    return processed_text

def recommend_job_categories(resume_text, model, vectorizer, label_encoder, top_n=3):
    processed_text = preprocess_text(resume_text)
    text_vector = vectorizer.transform([processed_text])
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_vector)[0]
        sorted_indices = probabilities.argsort()[::-1]
        recommendations = [(label_encoder.classes_[idx], probabilities[idx]) for idx in sorted_indices[:top_n]]
    else:
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(text_vector)[0]
            if decision_scores.ndim == 0:
                decision_scores = np.array([1 - decision_scores, decision_scores])
            sorted_indices = decision_scores.argsort()[::-1]
            recommendations = [(label_encoder.classes_[idx], decision_scores[idx]) for idx in sorted_indices[:top_n]]
        else:
            prediction = model.predict(text_vector)[0]
            recommendations = [(label_encoder.classes_[prediction], 1.0)]
    return recommendations

def extract_useful_sections(text):
    """
    Extracts useful sections from resume text: Skills, Experience, Education, Projects.
    Returns a string with only these sections if found.
    """
    # Normalize text
    lines = text.splitlines()
    useful_sections = []
    section_keywords = ['skill', 'experience', 'work experience', 'education', 'project', 'projects', 'certification', 'summary']
    current_section = None
    section_text = []

    for line in lines:
        line_lower = line.strip().lower()
        # Check if line is a section header
        if any(kw in line_lower for kw in section_keywords):
            if current_section and section_text:
                useful_sections.append(f"## {current_section.title()}\n" + "\n".join(section_text))
            current_section = line.strip()
            section_text = []
        elif current_section:
            section_text.append(line.strip())
    # Add last section
    if current_section and section_text:
        useful_sections.append(f"## {current_section.title()}\n" + "\n".join(section_text))

    # If nothing found, return original text
    if useful_sections:
        return "\n\n".join(useful_sections)
    else:
        return text

def summarize_text(text, max_length=130, min_length=30):
    # Hugging Face models have a max token limit (1024 for BART)
    if len(text.split()) > 500:
        text = " ".join(text.split()[:500])
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

st.title("Resume Recommendation System")

st.write("Upload your resume (PDF/DOCX) or paste your resume text below:")

uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
resume_text = ""

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''
        resume_text = text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = '\n'.join([para.text for para in doc.paragraphs])
        resume_text = text
    else:
        st.error("Unsupported file type.")
    # Extract only useful sections
    useful_text = extract_useful_sections(resume_text)
    st.write("Extracted Useful Resume Information:")
    if st.button("Summarize Extracted Information"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(useful_text)
        st.write("**Summary:**")
        st.success(summary)
    resume_text = st.text_area("Resume Text", useful_text, height=300)
else:
    resume_text = st.text_area("Or paste your resume text here", "", height=200)

if st.button("Get Recommendations"):
    if resume_text.strip() == "":
        st.warning("Please upload a resume or enter text.")
    else:
        processed = preprocess_text(resume_text)
        features = vectorizer.transform([processed])
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(features)[0]
            top_indices = np.argsort(preds)[::-1][:3]
        else:
            scores = model.decision_function(features)[0]
            top_indices = np.argsort(scores)[::-1][:3]
        top_categories = label_encoder.inverse_transform(top_indices)
        st.success("Top Recommended Job Categories:")
        for i, cat in enumerate(top_categories, 1):
            st.write(f"{i}. {cat}")

st.markdown('---')
