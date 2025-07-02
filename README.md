# Resume Summarizer and Recommendation System

A comprehensive AI-powered system that analyzes resumes to provide job category recommendations and intelligent resume summarization. This project combines machine learning classification with natural language processing to help job seekers understand their career prospects and optimize their resumes.

## 🚀 Features

- **Resume Classification**: Automatically categorizes resumes into relevant job categories using multiple ML models
- **Intelligent Summarization**: Extracts and summarizes key sections (Skills, Experience, Education, Projects) using BART transformer model
- **Multi-format Support**: Handles both PDF and DOCX resume formats
- **Interactive Web Interface**: User-friendly Streamlit web application
- **Model Comparison**: Evaluates multiple ML algorithms to find the best performing model
- **Real-time Processing**: Fast prediction and summarization with cached models

## 📊 Project Structure

```
Resume Summarizer and Recommendation System/
├── README.md                              # Project documentation
├── Resume recommendation and summarize app.py  # Streamlit web application
├── resume_recommendation.ipynb            # Jupyter notebook for model development
├── resume_model.pkl                       # Trained classification model
├── tfidf_vectorizer.pkl                   # TF-IDF vectorizer for text processing
└── label_encoder.pkl                      # Label encoder for job categories
```

## 🛠️ Technologies Used

### Machine Learning & NLP
- **scikit-learn**: For machine learning models (Random Forest, SVM, Naive Bayes, Logistic Regression)
- **NLTK**: Natural language processing (tokenization, stopwords, lemmatization)
- **TF-IDF Vectorization**: Text feature extraction
- **Transformers (Hugging Face)**: BART model for text summarization

### Data Processing & Analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization

### Web Framework & File Processing
- **Streamlit**: Interactive web application framework
- **pdfplumber**: PDF text extraction
- **python-docx**: DOCX file processing
- **pickle**: Model serialization

### Development Tools
- **Jupyter Notebook**: Model development and experimentation
- **kagglehub**: Dataset management

## 📋 Requirements

```
streamlit
scikit-learn
nltk
pandas
numpy
matplotlib
seaborn
transformers
torch
pdfplumber
python-docx
kagglehub
pickle-mixin
```

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/resume-summarizer-recommendation-system.git
   cd resume-summarizer-recommendation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (will be downloaded automatically on first run)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 🚀 Usage

### Web Application

1. **Run the Streamlit app**
   ```bash
   streamlit run "Resume recommendation and summarize app.py"
   ```

2. **Use the application**
   - Upload a resume file (PDF or DOCX) or paste resume text
   - Get intelligent summarization of key resume sections
   - Receive top 3 job category recommendations with confidence scores

### Jupyter Notebook Development

1. **Open the notebook**
   ```bash
   jupyter notebook resume_recommendation.ipynb
   ```

2. **Run all cells** to:
   - Load and analyze the resume dataset
   - Preprocess text data
   - Train multiple ML models
   - Evaluate model performance
   - Save trained models

## 🤖 Model Performance

The system evaluates multiple machine learning algorithms:

| Model | Description | Use Case |
|-------|-------------|----------|
| **Random Forest** | Ensemble method with decision trees | Robust classification with feature importance |
| **Linear SVM** | Support Vector Machine with linear kernel | High-dimensional text classification |
| **Naive Bayes** | Probabilistic classifier | Fast text classification |
| **Logistic Regression** | Linear probabilistic classifier | Interpretable classification |

The best performing model is automatically selected and saved for the web application.

## 📊 Data Processing Pipeline

1. **Text Preprocessing**
   - Convert to lowercase
   - Remove URLs, emails, and special characters
   - Tokenization using NLTK
   - Remove stopwords
   - Lemmatization
   - Filter short words

2. **Feature Extraction**
   - TF-IDF vectorization
   - N-gram analysis
   - Document-term matrix creation

3. **Model Training**
   - Train-test split (80-20)
   - Cross-validation
   - Hyperparameter optimization
   - Model evaluation and comparison

## 🎯 Key Features Explained

### Resume Summarization
- **Section Extraction**: Automatically identifies Skills, Experience, Education, Projects sections
- **BART Summarization**: Uses Facebook's BART-large-CNN model for high-quality summaries
- **Content Filtering**: Focuses on most relevant information for job applications

### Job Category Recommendation
- **Multi-class Classification**: Predicts multiple relevant job categories
- **Confidence Scores**: Provides probability/confidence for each recommendation
- **Top-N Recommendations**: Returns the most likely job categories

### Text Processing
- **Robust Preprocessing**: Handles various resume formats and layouts
- **NLP Pipeline**: Complete text normalization and feature extraction
- **Scalable Architecture**: Efficient processing for batch operations

## 📈 Future Enhancements

### Planned Features
- [ ] **Skill Gap Analysis**: Compare resume skills with job requirements
- [ ] **Resume Improvement Suggestions**: AI-powered recommendations for resume enhancement
- [ ] **Job Matching**: Direct matching with real job postings
- [ ] **ATS Optimization**: Resume formatting suggestions for Applicant Tracking Systems

### Technical Improvements
- [ ] **Advanced NLP**: Integration of BERT/RoBERTa for better semantic understanding
- [ ] **Deep Learning**: Neural network models for improved accuracy
- [ ] **Real-time Learning**: Model updates based on user feedback
- [ ] **Multi-language Support**: Support for resumes in different languages

## 🔍 Model Evaluation Metrics

The system tracks multiple performance metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for each job category
- **Recall**: Sensitivity for each job category
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification analysis

## 🛠️ Development

### Model Training Process
1. **Data Loading**: Import resume dataset from Kaggle
2. **Exploratory Data Analysis**: Understand data distribution and patterns
3. **Text Preprocessing**: Clean and normalize resume text
4. **Feature Engineering**: Create TF-IDF features
5. **Model Training**: Train multiple ML algorithms
6. **Model Evaluation**: Compare performance metrics
7. **Model Selection**: Choose best performing model
8. **Model Serialization**: Save models for production use

### Code Structure
- **Modular Design**: Separate functions for preprocessing, training, and prediction
- **Error Handling**: Robust error handling for file processing
- **Caching**: Streamlit caching for improved performance
- **Documentation**: Comprehensive inline documentation

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Resume dataset from [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- Hugging Face Transformers for BART summarization model
- Streamlit community for the amazing web framework
- NLTK contributors for natural language processing tools

## 📞 Support

If you have any questions or need help with the project, please:
1. Check the [Issues](https://github.com/yourusername/resume-summarizer-recommendation-system/issues) page
2. Create a new issue if your question isn't already addressed
3. Provide detailed information about your problem

---

**Note**: Make sure to replace `yourusername` with your actual GitHub username and add a `requirements.txt` file with all the dependencies listed above.
