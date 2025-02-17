#read a set of comments and have the models make decision if there was fraud or no.

#maybe be useful:            https://huggingface.co/model


# comments=[
#   {
#     "user_id": 1,
#     "user_name": "John Doe",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "The food was delivered on time and everything was fresh."
#   },
#   {
#     "user_id": 2,
#     "user_name": "Jane Smith",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "I waited the whole day but no one showed up. I didn't get any food."
#   },
#   {
#     "user_id": 3,
#     "user_name": "Michael Brown",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "Got my food but it was a little late. Everything was okay otherwise."
#   },
#   {
#     "user_id": 4,
#     "user_name": "Emily White",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "Food arrived on time and was in great condition."
#   },
#   {
#     "user_id": 5,
#     "user_name": "Chris Black",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "I was told the food would arrive by noon, but it never came."
#   },
#   {
#     "user_id": 6,
#     "user_name": "Laura Green",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "Food arrived late but still acceptable."
#   },
#   {
#     "user_id": 7,
#     "user_name": "Peter Wilson",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "Everything went smoothly. The food arrived as expected."
#   },
#   {
#     "user_id": 8,
#     "user_name": "Sophia Taylor",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "I didn't receive my food today."
#   },
#   {
#     "user_id": 9,
#     "user_name": "Matthew Adams",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "Food was delivered, but it was missing some items."
#   },
#   {
#     "user_id": 10,
#     "user_name": "Olivia Brown",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "Food arrived right on time. Everything was perfect."
#   },
#   {
#     "user_id": 11,
#     "user_name": "Nathan King",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "Food was a bit delayed, but I did get it."
#   },
#   {
#     "user_id": 12,
#     "user_name": "Ava Scott",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "I waited for hours but still no food. This is the second time this happened."
#   },
#   {
#     "user_id": 13,
#     "user_name": "Daniel Harris",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "Got my food on time. Thanks!"
#   },
#   {
#     "user_id": 14,
#     "user_name": "Isabella Clark",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "The delivery never arrived today. I'm very disappointed."
#   },
#   {
#     "user_id": 15,
#     "user_name": "James Lewis",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "The food was delivered a bit late, but I received it eventually."
#   },
#   {
#     "user_id": 16,
#     "user_name": "Mia Young",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "Everything was good. The food arrived on time."
#   },
#   {
#     "user_id": 17,
#     "user_name": "Alexander Wright",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "Food came a little late, but everything else was fine."
#   },
#   {
#     "user_id": 18,
#     "user_name": "Amelia Turner",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "No issues. Food was delivered as promised."
#   },
#   {
#     "user_id": 19,
#     "user_name": "Lucas Hill",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "Didn't receive my food. I'm unsure what went wrong."
#   },
#   {
#     "user_id": 20,
#     "user_name": "Charlotte Martin",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "Food was delivered, but there was some delay."
#   },
#   {
#     "user_id": 21,
#     "user_name": "Mason Rodriguez",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "Everything was delivered on time. The service was great."
#   },
#   {
#     "user_id": 22,
#     "user_name": "Ella Harris",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "The delivery was late, but I did get my food."
#   },
#   {
#     "user_id": 23,
#     "user_name": "Logan Edwards",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "No food delivery today. I waited but nothing arrived."
#   },
#   {
#     "user_id": 24,
#     "user_name": "Grace Walker",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "Got my food on time. No issues."
#   },
#   {
#     "user_id": 25,
#     "user_name": "Henry Carter",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "Late delivery but food arrived eventually."
#   },
#   {
#     "user_id": 26,
#     "user_name": "Scarlett Parker",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "Food was delivered right on time."
#   },
#   {
#     "user_id": 27,
#     "user_name": "Jackson Bennett",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "The food delivery never arrived. Not sure what happened."
#   },
#   {
#     "user_id": 28,
#     "user_name": "Liam Ward",
#     "delivery_status": "received",
#     "distributor": "Paul",
#     "comment": "I got my food but some items were missing."
#   },
#   {
#     "user_id": 29,
#     "user_name": "Ethan Phillips",
#     "delivery_status": "received",
#     "distributor": "Mark",
#     "comment": "No complaints. The delivery was smooth and on time."
#   },
#   {
#     "user_id": 30,
#     "user_name": "Zoe Mitchell",
#     "delivery_status": "not_received",
#     "distributor": "Sarah",
#     "comment": "Still no food delivery today. This is a recurring issue."
#   }
# ]




# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC  # Example: Support Vector Classifier for classification

class CommentDataLoader:
    def __init__(self, csv_file_path):
        """
        Initializes the loader with the path to the CSV file.
        """
        self.csv_file_path = csv_file_path
        self.comments = []

    def load_data(self):
        """
        Reads the comments data from the CSV file and loads it into memory.
        """
        # Load CSV data (e.g., using pandas or csv module)
        pass

    def clean_data(self):
        """
        Cleans the comment data (e.g., removes special characters, converts to lowercase).
        """
        # Perform basic text cleaning such as removing special characters, stopwords, etc.
        pass


class FraudDetectionModel:
    def __init__(self):
        """
        Initializes the fraud detection model.
        """
        self.vectorizer = TfidfVectorizer()  # TF-IDF for text representation
        self.model = SVC()  # Example: Support Vector Classifier for fraud detection

    def preprocess_comments(self, comments):
        """
        Preprocesses and vectorizes the comments using TF-IDF or another suitable NLP technique.
        """
        # Use TF-IDF or Word2Vec/fastText to convert text into feature vectors
        pass

    def train_model(self, X_train, y_train):
        """
        Trains the fraud detection model using labeled data.
        """
        # Train the SVM (or another classifier such as Logistic Regression, Random Forest, etc.)
        pass

    def predict(self, X_test):
        """
        Predicts whether the comments indicate fraud based on the trained model.
        """
        # Use the trained model to predict fraud based on new comment data
        pass

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the performance of the fraud detection model.
        """
        # Use accuracy, precision, recall, F1-score for evaluation
        pass


class NLPProcessor:
    def __init__(self, comments):
        """
        Class responsible for processing natural language comments.
        """
        self.comments = comments
        self.processed_comments = []

    def tokenize(self):
        """
        Tokenizes the comments into words.
        """
        # Tokenize the text using nltk or spaCy tokenizer
        pass

    def remove_stopwords(self):
        """
        Removes common stopwords from the comments.
        """
        # Use nltk's stopword list to remove stopwords from the comments
        pass

    def lemmatize(self):
        """
        Lemmatizes the words in the comments.
        """
        # Use nltk or spaCy for lemmatization
        pass

    def extract_features(self):
        """
        Extracts relevant features from the text data (e.g., TF-IDF, word embeddings).
        """
        # Convert the text into feature vectors using TF-IDF, Word2Vec, or similar
        pass


class FraudDetectionPipeline:
    def __init__(self, csv_file_path):
        """
        High-level pipeline to handle the entire fraud detection process.
        """
        self.data_loader = CommentDataLoader(csv_file_path)
        self.nlp_processor = None
        self.fraud_model = FraudDetectionModel()

    def run_pipeline(self):
        """
        Executes the full fraud detection pipeline from loading data to model prediction.
        """
        # Step 1: Load and clean data
        self.data_loader.load_data()
        self.data_loader.clean_data()

        # Step 2: Initialize NLP processor and preprocess text
        self.nlp_processor = NLPProcessor(self.data_loader.comments)
        self.nlp_processor.tokenize()
        self.nlp_processor.remove_stopwords()
        self.nlp_processor.lemmatize()
        self.nlp_processor.extract_features()

        # Step 3: Train the model (you would need to split the data into training and test sets)
        X_train, y_train = None, None  # Example placeholder
        self.fraud_model.train_model(X_train, y_train)

        # Step 4: Predict and evaluate
        X_test = None  # Placeholder for test data
        self.fraud_model.predict(X_test)
        y_test = None  # Placeholder for test labels
        self.fraud_model.evaluate_model(X_test, y_test)


class ReportGenerator:
    def __init__(self, predictions):
        """
        Generates a report based on fraud detection predictions.
        """
        self.predictions = predictions

    def generate_report(self):
        """
        Generates and saves a fraud detection report.
        """
        # Logic to generate the report (e.g., save as CSV, JSON, or display)
        pass

    def display_report(self):
        """
        Displays the fraud detection results.
        """
        # Display the results to a console or a dashboard
        pass

import requests

import json 
def fetch(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    data = response.json()
    print(data)




# Main script flow
if __name__ == "__main__":
    # Step 1: Initialize the fraud detection pipeline with the CSV data
    # pipeline = FraudDetectionPipeline("comments_data.csv")

    # # Step 2: Run the pipeline to detect fraud based on comments
    # pipeline.run_pipeline()
    # fetch("https://")
