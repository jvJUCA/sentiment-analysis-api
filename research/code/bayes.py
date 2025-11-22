import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words('english')
import matplotlib.pyplot as plt
import random


from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

# Dictionary for Labeling
LABELS = {
    'Positive': 1,
    'Negative': 0,
    'Neutral': 2
}

def num_to_label(num):
    for label, number in LABELS.items():
        if num == number:
            return label

    return None


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        # First, we will convert the pos_tag output tags to a tag format that the WordNetLemmatizer can interpret
        # In general, if a tag starts with NN, the word is a noun and if it stars with VB, the word is a verb.
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def clean_comment(comment):
    clean_comment = []

    for word in comment:
        # Remove stop words
        if word.lower()  in STOP_WORDS:
            continue

        # Remove punctuation
        if not word.isalpha():
            continue

        clean_comment.append(word)



    return clean_comment

   
if __name__ == "__main__":

    # Read CSV file
    df = pd.read_csv("synthetic_reviews.csv")
    print(df.head())


    # print(df['Sentiment'].value_counts())


    
    data = []

    # Separating our features (text) and our labels into two lists to smoothen our work
    X = df['Comment'].tolist()
    Y = df['Sentiment'].tolist()
    # Converting the labels to binary values
    Y = [LABELS[label] for label in Y]

    # (1)Tokenizing each comment into words
    for comment in X:
        data.append(word_tokenize(comment))

    print(data[0:5])

    # (2)Lemitizing the comments
    # Previewing the pos_tag() output
    # print(pos_tag(data[1]))    
    # Previewing the WordNetLemmatizer() output

    for i in range(len(data)):
        data[i] = lemmatize_sentence(data[i])
    print(data[0:5])


    # (3) Cleaning the data
    cleaned_data = []




    for comment in data:
        cleaned_data.append(clean_comment(comment))
    print(cleaned_data[0:5])



    # As the Naive Bayesian classifier accepts inputs in a dict-like structure,
    # we have to define a function that transforms our data into the required input structure
    def list_to_dict(cleaned_tokens):
        return dict([token, True] for token in cleaned_tokens)
    

    final_data = []

    # Transforming the data to fit the input structure of the Naive Bayesian classifier
    for tokens, label in zip(cleaned_data, Y):
        final_data.append((list_to_dict(tokens), label))

    print(final_data[0:5])


        
    # from wordcloud import WordCloud, STOPWORDS

    # positive_words = []
    # negative_words = []

    # # Separating out positive and negative words (i.e., words appearing in negative and positive tweets),
    # # in order to visualize each set of words seperately
    # for i in range(len(final_data)):
    #     if final_data[i][1] == 2:
    #         positive_words.extend(final_data[i][0])
    #     elif final_data[i][1] == 0:
    #         negative_words.extend(final_data[i][0])


    # # Defining our word cloud drawing function
    # def wordcloud_draw(data, color = 'black'):
    #     wordcloud = WordCloud(stopwords = STOPWORDS,
    #                         background_color = color,
    #                         width = 2500,
    #                         height = 2000
    #                         ).generate(' '.join(data))
    #     plt.figure(1, figsize = (13, 13))
    #     plt.imshow(wordcloud)
    #     plt.axis('off')
    #     plt.show()
        
    # print("Positive words")
    # wordcloud_draw(positive_words, 'white')
    # print("Negative words")
    # wordcloud_draw(negative_words)        


    # .Random(140) randomizes our data with seed = 140. This guarantees the same shuffling for every execution of our code
    # Feel free to alter this value or even omit it to have different outputs for each code execution
    random.Random(42).shuffle(final_data)

    # Here we decided to split our data as 90% train data and 10% test data
    # Once again, feel free to alter this number and test the model accuracy
    trim_index = int(len(final_data) * 0.9)

    train_data = final_data[:trim_index]
    test_data = final_data[trim_index:]

    print(len(train_data))
    print(len(test_data))


    from nltk import classify
    from nltk import NaiveBayesClassifier
    classifier = NaiveBayesClassifier.train(train_data)


    # Output the model accuracy on the train and test data
    print('Accuracy on train data:', classify.accuracy(classifier, train_data))
    print('Accuracy on test data:', classify.accuracy(classifier, test_data))

    # Classify the test data

    for i in range(10):
        print('Comment:', test_data[i][0])
        # Mapping the predicted label to the original label
        print('Original Label:', num_to_label(test_data[i][1]))
        print('Predicted Label:', num_to_label(classifier.classify(test_data[i][0])))

        print('\n')

    # Splitting the data into training and testing data
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("Hello, World!")