import string
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import re
import random
import numpy as np
import joblib


#Load SpaCy
nlp = spacy.load("en_core_web_sm")
RE_D = re.compile(r'\d')

# Random seed
random.seed(43)

# Set Pandas display options
pd.options.display.max_columns = None
pd.options.display.max_rows = None

laplace_smoothing = 1
ham_class_label = "Ham Probability"
spam_class_label = "Spam Probability"

# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#
#     # Replace numbers with 'aanumbers'
#     # text = ' '.join(['aanumbers' if RE_D.search(item) else item for item in text.split(' ')])
#
#     # Remove punctuation
#     # text = text.translate(str.maketrans(' ', ' ', string.punctuation))
#     text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
#
#     # Process with SpaCy
#     doc = nlp(text)
#     processed_text = []
#     for token in doc:
#         # Remove stopwords and single letters
#         if token.text in STOP_WORDS or len(token.text) == 1:
#             continue
#
#         if RE_D.search(token.text):
#             processed_text.append('aanumbers')
#             continue
#
#         # Lemmatize
#         processed_text.append(token.lemma_)
#
#     return ' '.join(processed_text)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Process with SpaCy
    doc = nlp(text)

    # Lemmatization, remove stop words, and punctuation
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # Remove numbers and replace with 'aanumbers'
    processed_tokens = ['aanumbers' if re.search(r'\d', token) else token for token in processed_tokens]

    # Remove single letters
    processed_tokens = [token for token in processed_tokens if len(token) > 1]

    return " ".join(processed_tokens)


def bag_of_words(split_string, operation="train"):
    if not isinstance(split_string, pd.Series):
        raise TypeError("Input must be a pandas Series of strings")

    vectorizer = CountVectorizer()

    try:
        if operation == "train":
            X = vectorizer.fit_transform(split_string)

            # Save the trained vectorizer
            joblib.dump(vectorizer, "vectorizer.pkl")
        else:  # 'test'
            load_vectorizer = joblib.load("vectorizer.pkl")
            # Load the trained vectorizer
            vectorizer = joblib.load("vectorizer.pkl")

            # Manually set vocabulary to match training
            new_vectorizer = CountVectorizer(vocabulary=load_vectorizer.vocabulary_)

            # Transform new data using the fixed vocabulary
            X = new_vectorizer.transform(split_string)

        # Convert sparse matrix to DataFrame
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    except Exception as e:
        print(f"Error in bag_of_words function: {e}")
        return None

def cond_prob_calculator(n_word_class_i, n_class):
    return (n_word_class_i + laplace_smoothing) / (n_class + (laplace_smoothing * n_vocab))

def probability_calculator(phrase, class_type, prior_prob, n_class):
    word_arr = phrase.split(" ")
    prob = np.log(prior_prob)
    for word in word_arr:
        try:
            prob += np.log(cond_prob_df.at[word, class_type])
        except KeyError:
            prob += np.log(cond_prob_calculator(0.0, n_class))
    return prob


def classifier(phrase):
    ham_phrase_prob = probability_calculator(phrase, ham_class_label, p_ham, n_ham)
    spam_phrase_prob = probability_calculator(phrase, spam_class_label, p_spam, n_spam)

    if ham_phrase_prob > spam_phrase_prob:
        return 0
    elif ham_phrase_prob < spam_phrase_prob:
        return 1
    else:
        return "unknown"

# Load preprocessed ds from file if exist, else preprocess spam.csv dataset
try:
    df = pd.read_csv("./ds.csv", encoding='iso-8859-1', usecols=[0, 1], header=0, names=['Target', 'SMS'])
    df['SMS'] = df['SMS'].fillna("")
except FileNotFoundError:
    df = pd.read_csv('spam.csv', encoding='iso-8859-1', usecols=[0, 1], header=0, names=['Target', 'SMS'])
    for index, row in df.iterrows():
        df.at[index, 'SMS'] = preprocess_text(row['SMS'])
        df.at[index, 'Target'] = 0 if row["Target"] == 'ham' else 1
    # Remove empty strings
    # df = df[df['SMS'].str.strip() != '']

    df.to_csv('ds.csv', encoding='iso-8859-1', index=False)

# Randomize ds
df = df.sample(n=df.shape[0], random_state=43, ignore_index=False)

# Split dataset on train and test
train_last_index = int(df.shape[0] * 0.80)
train_set = df.iloc[:train_last_index]
test_set = df.iloc[train_last_index:]

# Create bag of words
train_bag_of_words = bag_of_words(train_set['SMS'])

# Overwrite pd display options
pd.options.display.max_columns = train_bag_of_words.shape[1]
pd.options.display.max_rows = train_bag_of_words.shape[0]

# Get vocabulary size
n_vocab = train_bag_of_words.shape[1]

# Add BoW to train ds
train_set = pd.concat([train_set.reset_index(drop=True), train_bag_of_words], axis=1)

# Calculate conditional prob df using train df
cond_prob_df = pd.DataFrame(0.0, index=train_set.columns[2:].values, columns=[spam_class_label, ham_class_label])

n_word_spam = train_set[train_set["Target"] == 1].iloc[:,2:].sum()
n_word_ham = train_set[train_set["Target"] == 0].iloc[:,2:].sum()

n_spam = n_word_spam.sum()  # Number of spam messages
n_ham = n_word_ham.sum()  # Number of ham messages

# Prior prob
p_spam = n_spam / (n_spam + n_ham)
p_ham = n_ham / (n_ham + n_spam)

# Make prediction using test ds
for index, row in cond_prob_df.iterrows():
    cond_prob_df.at[index, spam_class_label] = (cond_prob_calculator(n_word_spam[index], n_spam)).round(6)
    cond_prob_df.at[index, ham_class_label] =  (cond_prob_calculator(n_word_ham[index], n_ham)).round(6)

pred_df = pd.DataFrame({
    "Predicted": test_set["SMS"].apply(lambda x: classifier(x)),
    "Actual": test_set["Target"]
}, index=test_set.index)

# Creating confusion matrix
TP = len(pred_df[(pred_df["Predicted"] == 1) & (pred_df["Actual"] == 1)])
TN = len(pred_df[(pred_df["Predicted"] == 0) & (pred_df["Actual"] == 0)])
FP = len(pred_df[(pred_df["Predicted"] == 1) & (pred_df["Actual"] == 0)])
FN = len(pred_df[(pred_df["Predicted"] == 0) & (pred_df["Actual"] == 1)])

confusion_df = pd.DataFrame(
    [[TP, FN],
     [FP, TN]],
    columns=["Predicted Spam", "Predicted Ham"],
    index=["Actual Spam", "Actual Ham"]
)

precision = (TP / (TP + FP))
recall = (TP / (TP + FN))

# Test BoW
test_bag_of_words = bag_of_words(test_set['SMS'], "test")
test_set = pd.concat([test_set.reset_index(drop=True), test_bag_of_words], axis=1)

# create MultinomialNB model
model = MultinomialNB()
model.fit(train_bag_of_words, train_set['Target'])
predicted_test = model.predict(test_bag_of_words)

# Output dict
output = {
    'Accuracy': ((TP + TN) / (TP + TN + FP + FN)),
    'Recall': recall,
    'Precision': precision,
    'F1': 2 * ((precision * recall) / (precision + recall))
    }

output_model = {
    'Accuracy': accuracy_score(test_set['Target'], predicted_test),
    'Recall': recall_score(test_set['Target'], predicted_test),
    'Precision': precision_score(test_set['Target'], predicted_test),
    'F1': f1_score(test_set['Target'], predicted_test)
    }

print(output_model)