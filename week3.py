import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. LOAD DATA

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

# 2. DATA CLEANING

df = df.iloc[:, :2]

df.columns = ["label", "message"]

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

print("\nNull values:")
print(df.isnull().sum())

print("\nDataset shape:", df.shape)

# 3. LABEL DISTRIBUTION

print("\nMessage Labels:")
print(df["label"].unique())

print("\nLabel Counts:")
print(df["label"].value_counts())

# 4. LABEL COUNT - BAR CHART
 

label_counts = df["label"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    label_counts.index,
    label_counts.values
)

plt.title("Number of Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")

plt.show()

# 5. TEXT PREPROCESSING
 
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Convert messages to lowercase
df["message"] = df["message"].str.lower()

print("\nPreprocessed Messages:")
print(df.head())

# 6. SPLIT DATA INTO TRAINING AND TESTING
 
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 7. TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF Testing Shape:")
print(X_test_tfidf.shape)

# 8. TRAIN NAIVE BAYES CLASSIFIER
 
model = MultinomialNB()

model.fit(
    X_train_tfidf,
    y_train
)

print("\nNaive Bayes model trained successfully.")

# 9. PREDICTION
y_pred = model.predict(X_test_tfidf)
 
# 10. MODEL ACCURACY

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)
# 11. CLASSIFICATION REPORT
 
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)


 
# 12. CONFUSION MATRIX
 

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Ham", "Spam"]
)

plt.yticks(
    [0, 1],
    ["Ham", "Spam"]
)
 
for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.show()
 
# 13. TEST WITH NEW MESSAGES

new_messages = [
    "Congratulations! You have won a free prize. Call now!",
    "Hi, are you coming to college today?",
    "You have won 100000 dollars. Claim your prize now!",
    "Can you send me the assignment?"
]

# Convert new messages using the same TF-IDF vectorizer
new_messages_tfidf = vectorizer.transform(new_messages)

 
predictions = model.predict(new_messages_tfidf)

print("\nNew Message Predictions:")

for message, prediction in zip(new_messages, predictions):

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    print("\nMessage:", message)
    print("Prediction:", result)