import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample data
data = pd.DataFrame({
    'email': ["Urgent: Meeting at 3 PM", "Discount on your next purchase", "Project deadline extended"],
    'category': ["urgent", "promotion", "update"]
})

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(data['email'], data['category'])

# Function to categorize email
def categorize_email(email_content):
    return model.predict([email_content])[0]

class EmailProcessor:
    def __init__(self):
        self.model = model

    def process_email(self, email_content):
        category = categorize_email(email_content)
        print(f"Email categorized as: {category}")
        # Additional logic based on the category

# Example usage with multiple test cases
email_processor = EmailProcessor()
test_emails = [
    "Urgent: Please respond ASAP",
    "Get 50% off on your next purchase",
    "The project deadline has been extended to next week"
]

for email in test_emails:
    email_processor.process_email(email)