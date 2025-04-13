# Fake News Detection using Logistic Regression

This project demonstrates a simple machine learning model to detect fake news articles using Logistic Regression. The dataset used is from Kaggle and consists of news article metadata (title, author) and labels (real/fake).
## Dataset
- **LINK** https://www.kaggle.com/c/fake-news/data?select=train.csv

## Dependencies

- **pandas**: For handling the dataset (CSV)
- **numpy**: For numerical operations
- **re**: For regex-based text cleaning
- **nltk**: Natural Language Toolkit for stopwords and stemming
- **sklearn**:
  - `TfidfVectorizer` for converting text into vectors
  - `train_test_split` for splitting the dataset
  - `LogisticRegression` as the model
  - `accuracy_score` for evaluation

```bash
pip install numpy pandas nltk scikit-learn
```

## Dataset
Each entry in the dataset contains:
- `id`: Unique identifier
- `title`: Title of the article
- `author`: Author name
- `text`: News content (incomplete in some cases)
- `label`: 0 for real, 1 for fake

### Preprocessing
- Fill missing values with empty strings
- Merge `title` and `author` columns to form a new feature called `content`
- Use regex to clean the text and remove all characters except alphabets
- Convert all words to lowercase
- Remove stopwords (e.g., "the", "is")
- Apply stemming (e.g., "running" -> "run")

```python
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)
```

## Vectorization
Use `TfidfVectorizer` to convert the cleaned text data into numerical form.
```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news_data['content'].values)
```

## Splitting Data
Split the dataset into training and testing sets (80/20 split).
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

## Training the Model
```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## Evaluation
```python
training_data_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(model.predict(X_test), Y_test)
```

### Sample Output
```
Accuracy score of the training data : 0.985
Accuracy score of the test data : 0.982
```

## Real-time Prediction System
For new input data:
- Apply the same preprocessing
- Use the trained vectorizer
- Predict using the model

```python
input_data = "ISRO successfully launched Chandrayaan-3 to the moon."
processed_input = preprocess_text(input_data)
vectorized_input = vectorizer.transform([processed_input])
prediction = model.predict(vectorized_input)
```

## Limitations
- The model was trained only on `title` and `author`, not full article text
- Generalization may be poor on real-world articles
- Logistic Regression is a basic model, better results may be obtained with advanced models like BERT

## Next Steps
- Improve input features (include full `text`)
- Use advanced models (e.g., Transformers)
- Explainable AI: Show *why* a news is fake

---
Made with love by [Your Name] for learning NLP and ML fundamentals.
