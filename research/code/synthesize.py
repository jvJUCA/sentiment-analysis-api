# import random
# import pandas as pd

# # Define features and sentiment labels
# features = [
#     "Login Process", "Navigation", "Search Functionality", "Product Pages",
#     "Checkout Process", "Customer Support", "Mobile Responsiveness", "Loading Times"
# ]

# sentiments = ["Positive", "Negative", "Neutral"]

# # Templates for comments
# templates = {
#     "Positive": [
#         "The {feature} is very user-friendly and efficient.",
#         "I love how smooth the {feature} is.",
#         "The {feature} exceeded my expectations.",
#         "{feature} works perfectly!",
#         "I'm very satisfied with the {feature}."
#     ],
#     "Negative": [
#         "I found the {feature} to be confusing and frustrating.",
#         "The {feature} needs a lot of improvement.",
#         "I'm disappointed with the {feature}.",
#         "{feature} is too slow and unresponsive.",
#         "The {feature} didn't work as I expected."
#     ],
#     "Neutral": [
#         "The {feature} works as expected.",
#         "{feature} is okay, nothing special.",
#         "I'm indifferent about the {feature}.",
#         "{feature} is just average.",
#         "The {feature} is neither good nor bad."
#     ]
# }

# # Generate synthetic comments
# def generate_comments(features, sentiments, templates, num_comments=100):
#     data = {"Comment": [], "Feature": [], "Sentiment": []}
#     for _ in range(num_comments):
#         feature = random.choice(features)
#         sentiment = random.choice(sentiments)
#         comment = random.choice(templates[sentiment]).format(feature=feature)
#         data["Comment"].append(comment)
#         data["Feature"].append(feature)
#         data["Sentiment"].append(sentiment)
#     return pd.DataFrame(data)

# # Generate the dataset
# num_comments = 200  # Specify the number of synthetic comments you want
# df = generate_comments(features, sentiments, templates, num_comments)

# # Display the first few rows of the dataset
# print(df.head())


# # Save the dataset to a CSV file
# df.to_csv("synthetic_reviews.csv", index=False)
import random
import pandas as pd
from nltk.corpus import wordnet

# Ensure NLTK wordnet data is downloaded
import nltk
nltk.download('wordnet')

# Function to get synonyms
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms) if synonyms else [word]

# Define features and sentiment labels
features = {
    "Login Process": ["Password Reset", "Two-Factor Authentication"],
    "Navigation": ["Menu Structure", "Breadcrumbs"],
    "Search Functionality": ["Search Speed", "Search Accuracy"],
    "Product Pages": ["Image Quality", "Product Descriptions"],
    "Checkout Process": ["Payment Methods", "Order Summary"],
    "Customer Support": ["Response Time", "Helpfulness"],
    "Mobile Responsiveness": ["Layout", "Touch Interactions"],
    "Loading Times": ["Page Load Speed", "Resource Load Speed"]
}

sentiments = ["Positive", "Negative", "Neutral"]

# Define words to replace with synonyms
synonym_words = {
    "user-friendly": ["user-friendly", "intuitive", "easy-to-use", "convenient", "accessible"],
    "efficient": ["efficient", "productive", "effective", "streamlined", "time-saving"],
    "confusing": ["confusing", "complicated", "bewildering", "perplexing", "unclear"],
    "frustrating": ["frustrating", "annoying", "irritating", "exasperating", "disappointing"],
    # Add synonyms for other contextually relevant words
}

# Generate synthetic comments with synonyms
def generate_comments(features, sentiments, synonym_words, num_comments=100):
    data = {"Comment": [], "Feature": [], "Sentiment": []}
    for _ in range(num_comments):
        main_feature = random.choice(list(features.keys()))
        sub_feature = random.choice(features[main_feature])
        feature = f"{main_feature} - {sub_feature}"
        sentiment = random.choice(sentiments)
        
        # Select random words from synonym_words for the current sentiment
        words = synonym_words[sentiment]
        
        # Fetch synonyms for each word and replace placeholders in the template
        synonyms = [random.choice(get_synonyms(word)) for word in words]
        comment = f"The {feature} is very {synonyms[0]} and {synonyms[1]}."
        
        data["Comment"].append(comment)
        data["Feature"].append(feature)
        data["Sentiment"].append(sentiment)
    return pd.DataFrame(data)

# Generate the dataset
num_comments = 1000  # Specify the number of synthetic comments you want
df = generate_comments(features, sentiments, synonym_words, num_comments)

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('synthetic_usability_comments_with_synonyms.csv', index=False)
