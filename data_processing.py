import pandas as pd
from bs4 import BeautifulSoup

def process_html_str(raw_html):
    """
    Use BeautifulSoup to process raw 
    html and extract text.
    
    Args: 
    (str) raw_html: raw string of HTML ripe for BS4 parsing
    Returns:
    (str) text: raw text extracted from HTML
    """
    soup = BeautifulSoup(raw_html)
    return soup.get_text(' ', True)

def main(filepath):
    df = pd.read_json(filepath, orient='records', encoding_errors='replace')
    df['question'] = df['question'].apply(process_html_str)
    df['answer'] = df['answer'].apply(process_html_str)
    df.to_csv('preprocessing_text.json')

if __name__=='__main__':
    main('data/preprocessing_html_data.json')