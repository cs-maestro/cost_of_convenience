import csv
import re
from langdetect import detect_langs
from collections import defaultdict

def is_valid_text(text):
    # Example: Check if the text contains at least one alphabetic character
    return any(c.isalpha() for c in text)

def count_messages_by_language(csv_file):
    lang_count = defaultdict(int)

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        
        for row in reader:
            if len(row) > 4:  # Assuming the message is in the 5th column (index 4)
                message = row[4].strip()
                if is_valid_text(message):
                    try:
                        lang = detect_langs(message)
                        if lang:
                            lang_count[lang[0].lang] += 1
                    except Exception as e:
                        print(f"Error detecting language for message: {message}. Error: {str(e)}")

    return lang_count

# Example usage:
csv_file = './unique_url_messages.csv' 
language_counts = count_messages_by_language(csv_file)

# Print results
for lang, count in language_counts.items():
    print(f"Language: {lang}, Messages: {count}")
