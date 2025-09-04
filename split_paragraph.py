import spacy
import sys
import json

def split(text, output_path):
    nlp=None
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    with open(output_path, 'w') as file:
        json.dump(sentences, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    text = sys.argv[1]
    output_path = sys.argv[2]
    split(text, output_path)