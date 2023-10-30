from flask import Flask, request, render_template
from flask import jsonify
from flask_cors import CORS
import emoji
from nltk.corpus import words
import pyphen
import joblib
import nltk
import pandas as pd
import numpy as np

nltk.download('words')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_inp = request.form.get("domain")
        model = joblib.load("ENS_App_model.pkl")

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



        user_inp = str(user_inp)
        user_inp = user_inp.replace(" ", "").lower()
        user_inp = user_inp.split(".", 1)[0]

        x_data = pd.DataFrame()

        x_data['Domain'] = [user_inp]

        x_data['Domain'] = x_data['Domain'].str.replace(' ', '')

        x_data['Length'] = x_data['Domain'].apply(len)

        x_data = x_data.drop_duplicates(subset='Domain', keep='first')

        x_data['Repetition'] = x_data['Domain'].apply(has_repetition)

        x_data['IsAllDigits'] = x_data['Domain'].apply(is_all_digits)

        x_data['IsMixedDigitsLetters'] = x_data['Domain'].apply(is_mixed_digits_letters)

        x_data['StartsWithUnderscore'] = x_data['Domain'].apply(starts_with_underscore)

        x_data['IsPalindrome'] = x_data['Domain'].apply(is_palindrome)

        x_data['IsMirrorPalindrome'] = x_data['Domain'].apply(is_mirror_palindrome)

        x_data['IsEnglishWord'] = x_data['Domain'].apply(is_english_word)

        x_data['SyllableCount'] = x_data['Domain'].apply(count_syllables)

        x_data['ContainsEmoji'] = x_data['Domain'].apply(contains_emoji)

        x_data['ContainsEnglishSubset'] = x_data['Domain'].apply(contains_english_subset)

        boolean_columns = ['Repetition', 'IsAllDigits', 'IsMixedDigitsLetters',
                           'StartsWithUnderscore', 'IsPalindrome', 'IsMirrorPalindrome',
                           'IsEnglishWord', 'ContainsEmoji', 'ContainsEnglishSubset']

        for column in boolean_columns:
            x_data[column] = x_data[column].astype(int)

        print(x_data)

        prediction = model.predict(x_data.drop(columns=["Domain"], axis=1))


        def adjust_value(features, predicted_price):
            # Extract feature values
            length, repetition, is_all_digits, is_mixed_digits_letters, starts_with_underscore, \
                is_palindrome, is_mirror_palindrome, is_english_word, syllable_count, contains_emoji, contains_english_subset = features

            if length > 4:
                predicted_price *= 0.8
            if length > 8:
                predicted_price *= 0.5
            if length > 12:
                predicted_price *= 0.2
            if length <= 4 and predicted_price < 15:
                predicted_price *= 2
            if length > 12 and contains_english_subset == False:
                predicted_price *= 0.5
            if repetition and predicted_price < 15:
                predicted_price *= 1.2
            if is_all_digits:
                predicted_price *= 0.5
            if is_mixed_digits_letters and predicted_price < 15:
                predicted_price *= 1.2
            if starts_with_underscore:
                predicted_price *= 0.3
            if is_palindrome and predicted_price < 15:
                predicted_price *= 2
            if is_mirror_palindrome and predicted_price < 15:
                predicted_price *= 2
            if is_english_word and predicted_price < 15:
                predicted_price *= 2
            if is_english_word == False:
                predicted_price *= 0.8
            if syllable_count > 3:
                predicted_price *= 0.8
            if syllable_count <= 3 and predicted_price < 15:
                predicted_price *= 2
            if contains_emoji and predicted_price < 15:
                predicted_price *= 1.3
            if contains_english_subset and predicted_price < 15:
                predicted_price *= 2

            return predicted_price

        domain_features = x_data.drop(columns=["Domain"], axis=1)
        domain_features = np.array(domain_features)[0].tolist()
        predicted_listing_price = prediction

        adjusted_price = adjust_value(domain_features, predicted_listing_price)
        print(f'Adjusted Price: {adjusted_price}')

        appraisal_result = adjusted_price.tolist()
        

        return jsonify(appraisal=appraisal_result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
