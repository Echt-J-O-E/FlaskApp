from flask import Flask, request, render_template
from flask import jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import emoji
from nltk.corpus import words
import pyphen
import joblib
import nltk
nltk.download('words')

from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import cmudict
import re
from names_dataset import NameDataset


nltk.download('cmudict')
pronouncing_dict = cmudict.dict()


def appraise(domain_name):
    base_model = joblib.load("base_model.pkl")
    model_999 = joblib.load("model_999.pkl")
    model_10k = joblib.load("model_10k.pkl")
    model_100k = joblib.load("model_100k.pkl")
    model_3_letter_pal = joblib.load("model_3_letter_pal.pkl")
    model_5_digits_pal = joblib.load("model_5_digits_pal.pkl")
    model_alphanumeric_3 = joblib.load("model_alphanumeric_3.pkl")
    model_alphanumeric_4 = joblib.load("model_alphanumeric_4.pkl")
    model_letters_4 = joblib.load("model_letters_4.pkl")
    model_surnames = joblib.load("model_surnames.pkl")

    def has_repetition(domain):
        domain = domain.lower()
        for i in range(len(domain) - 1):
            if domain[i] == domain[i + 1]:
                return True
        return False

    def is_all_digits(domain):
        return domain.isdigit()

    def is_mixed_digits_letters(domain):
        return any(char.isdigit() for char in domain) and any(char.isalpha() for char in domain)

    def starts_with_underscore(domain):
        return domain.startswith('_')

    def is_palindrome(domain):
        domain = domain.replace(" ", "").lower()
        return domain == domain[::-1]

    def is_mirror_palindrome(domain):
        mirror_chars = {
            'A': 'A', 'B': ' ', 'C': ' ', 'D': ' ', 'E': '3',
            'F': ' ', 'G': ' ', 'H': 'H', 'I': 'I', 'J': 'L',
            'K': ' ', 'L': 'J', 'M': 'M', 'N': ' ', 'O': 'O',
            'P': ' ', 'Q': ' ', 'R': ' ', 'S': '2', 'T': 'T',
            'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': '5',
            '1': '1', '2': 'S', '3': 'E', '4': ' ', '5': 'Z',
            '6': ' ', '7': ' ', '8': '8', '9': ' ', '0': 'O'
        }

        mirror_domain = ''.join([mirror_chars.get(char, char) for char in domain])[::-1]

        return mirror_domain == domain

    english_words = set(words.words())

    def is_english_word(domain):
        return domain.lower() in english_words

    dic = pyphen.Pyphen(lang='en')

    def count_syllables(domain):
        syllable_count = sum(len(dic.positions(word)) + 1 for word in domain.split())
        return syllable_count

    def contains_emoji(domain):
        return emoji.emoji_count(domain) > 0

    def contains_english_subset(domain):
        domain = domain.lower()
        for word in english_words:
            if len(word) >= 3 and word in domain:
                return True
        return False

    def phonetic_appeal(domain):
        words = re.sub(r'[^a-zA-Z]', ' ', domain).split()
        phonetic_score = 0
        for word in words:
            if word.lower() in pronouncing_dict:
                phonetic_score += 1
        return phonetic_score / len(words) if words else 0

    def visual_simplicity(domain):
        return 1 if re.match(r'^[a-zA-Z]+$', domain) else 0

    def check_brandability(domain):
        memorability_score = (phonetic_appeal(domain) + visual_simplicity(domain) + (1 / len(domain))) / 3

        pronounceability_score = TextBlob(domain).sentiment.polarity

        relevant_keywords = ['tech', 'crypto', 'ai', 'nft']
        relevance_score = any(keyword in domain for keyword in relevant_keywords)

        response = requests.get(f"https://www.google.com/search?q={domain}")
        soup = BeautifulSoup(response.text, 'html.parser')
        uniqueness_score = len(soup.find_all(text=domain)) == 0  # No results found means more unique

        seo_potential_score = any(keyword in domain for keyword in relevant_keywords)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([domain] + relevant_keywords)
        keyword_importance_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten().max()

        brandability_score = (memorability_score + pronounceability_score +
                              relevance_score + uniqueness_score +
                              seo_potential_score + keyword_importance_score) / 6

        return brandability_score

    def check_memorability(domain):
        memorability_score = (phonetic_appeal(domain) + visual_simplicity(domain) + (1 / len(domain))) / 3
        return memorability_score

    def check_sentiment(domain):
        analysis = TextBlob(domain)
        return analysis.sentiment.polarity

    nd = NameDataset()

    def is_surname(word):
        info = nd.search(word)
        return info.get('last_name') is not None

    str(domain_name)
    user_dom = pd.DataFrame()

    user_dom['domain'] = [domain_name]

    user_dom['domain'] = user_dom['domain'].astype(str)

    user_dom['domain'] = user_dom['domain'].str.replace(' ', '')
    user_dom['Length'] = user_dom['domain'].apply(len)
    user_dom = user_dom.drop_duplicates(subset='domain', keep='first')
    user_dom.head()

    user_dom['Repetition'] = user_dom['domain'].apply(has_repetition)

    user_dom['IsAllDigits'] = user_dom['domain'].apply(is_all_digits)

    user_dom['IsMixedDigitsLetters'] = user_dom['domain'].apply(is_mixed_digits_letters)

    user_dom['StartsWithUnderscore'] = user_dom['domain'].apply(starts_with_underscore)

    user_dom['IsPalindrome'] = user_dom['domain'].apply(is_palindrome)

    user_dom['IsMirrorPalindrome'] = user_dom['domain'].apply(is_mirror_palindrome)

    user_dom['IsEnglishWord'] = user_dom['domain'].apply(is_english_word)

    user_dom['SyllableCount'] = user_dom['domain'].apply(count_syllables)

    user_dom['ContainsEmoji'] = user_dom['domain'].apply(contains_emoji)

    user_dom['ContainsEnglishSubset'] = user_dom['domain'].apply(contains_english_subset)

    boolean_columns = ['Repetition', 'IsAllDigits', 'IsMixedDigitsLetters',
                       'StartsWithUnderscore', 'IsPalindrome', 'IsMirrorPalindrome',
                       'IsEnglishWord', 'ContainsEmoji', 'ContainsEnglishSubset']

    for column in boolean_columns:
        user_dom[column] = user_dom[column].astype(int)

    user_dom['Brandability_Score'] = user_dom['domain'].apply(check_brandability)
    user_dom['Memorability_Score'] = user_dom['domain'].apply(check_memorability)
    user_dom['Sentiment_Score'] = user_dom['domain'].apply(check_sentiment)

    ### Predictions

    if len(domain_name) == 3 and domain_name.isdigit():
        prediction = model_999.predict(user_dom.drop(columns=["domain"], axis=1))

    elif len(domain_name) == 4 and domain_name.isdigit():
        prediction = model_10k.predict(user_dom.drop(columns=["domain"], axis=1))

    elif is_palindrome(domain_name) and domain_name.isdigit() and len(domain_name) == 5:
        prediction = model_5_digits_pal.predict(user_dom.drop(columns=["domain"], axis=1))

    elif len(domain_name) == 5 and domain_name.isdigit():
        prediction = model_100k.predict(user_dom.drop(columns=["domain"], axis=1))

    elif is_surname(domain_name):
        prediction = model_surnames.predict(user_dom.drop(columns=["domain"], axis=1))

    elif len(domain_name) == 4 and domain_name.isalpha():
        prediction = model_letters_4.predict(user_dom.drop(columns=["domain"], axis=1))

    elif is_palindrome(domain_name) and domain_name.isalpha() and len(domain_name) == 3:
        prediction = model_3_letter_pal.predict(user_dom.drop(columns=["domain"], axis=1))

    elif is_mixed_digits_letters(domain_name) and len(domain_name) == 3:
        prediction = model_alphanumeric_3.predict(user_dom.drop(columns=["domain"], axis=1))

    elif is_mixed_digits_letters(domain_name) and len(domain_name) == 4:
        prediction = model_alphanumeric_4.predict(user_dom.drop(columns=["domain"], axis=1))

    else:
        prediction = base_model.predict(user_dom.drop(columns=["domain"], axis=1))

    model_pred = prediction

    def adjust_value(features, predicted_price):
        # Extract feature values
        length, repetition, is_all_digits, is_mixed_digits_letters, starts_with_underscore, \
            is_palindrome, is_mirror_palindrome, is_english_word, syllable_count, contains_emoji, contains_english_subset, \
            brandability_score, memorability_score, sentiment_score = features

        if length > 4:
            predicted_price *= 0.8
        if length > 8:
            predicted_price *= 0.7
        if length > 15:
            predicted_price *= 0.5
        if length <= 4:
            predicted_price *= 1.2
        if length > 10 and contains_english_subset == False:
            predicted_price *= 0.5
        # if repetition:
        #     predicted_price *= 1.2
        if is_all_digits and length > 10:
            predicted_price *= 0.5
        if is_mixed_digits_letters and predicted_price < 15:
            predicted_price *= 1.2
        if starts_with_underscore:
            predicted_price *= 0.8
        if is_palindrome:
            predicted_price *= 1.2
        if is_mirror_palindrome:
            predicted_price *= 1.2
        if is_english_word:
            predicted_price *= 1.2
        if is_english_word == False:
            predicted_price *= 0.7
        if syllable_count > 3:
            predicted_price *= 0.8
        if syllable_count <= 3:
            predicted_price *= 1.2
        if contains_emoji and predicted_price < 15:
            predicted_price *= 1.2
        if contains_english_subset:
            predicted_price *= 1.2
        if sentiment_score > 0.5:
            predicted_price *= 1.2

        weight_memorability = 0.5  # Weight for memorability
        weight_brandability = 1  # Weight for brandability

        # Calculate the adjustment factor
        adjustment_factor = (weight_memorability * memorability_score) + (weight_brandability * brandability_score)

        predicted_price = predicted_price * (1 + adjustment_factor)

        return predicted_price

    domain_features = user_dom.drop(columns=["domain"], axis=1)
    domain_features = np.array(domain_features)[0].tolist()
    keys = ["length", "repetition", "is_all_digits", "is_mixed_digits_letters", "starts_with_underscore", \
            "is_palindrome", "is_mirror_palindrome", "is_english_word", "syllable_count", "contains_emoji",
            "contains_english_subset", \
            "brandability_score", "memorability_score", "sentiment_score"]

    features_dict = dict(zip(keys, domain_features))
    predicted_listing_price = prediction

    adjusted_price = adjust_value(domain_features, predicted_listing_price)
    # print(f'Adjusted Price: {adjusted_price}')

    appraisal_result = adjusted_price.tolist()

    # print("initial prediction", model_pred)
    #
    # print("adjusted prediction", appraisal_result)

    return appraisal_result, features_dict


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_inp = request.form.get("domain")
        user_inp = str(user_inp)
        user_inp = user_inp.replace(" ", "").lower()
        user_inp = user_inp.split(".", 1)[0]

        result = appraise(user_inp)

        return jsonify({
            'prediction': result[0],
            'features': result[1]
        })
    return "Appraiser is Active"



if __name__ == "__main__":
    app.run(debug=True)