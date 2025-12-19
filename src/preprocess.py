import re

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'[^a-z0-9 +\-*/<>=]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
