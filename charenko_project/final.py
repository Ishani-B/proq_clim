import os
import re
import pandas as pd
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import logging
import time

logging.basicConfig(level=logging.INFO)

TIMEOUT = 360
MAX_WORKERS = 4

author_pattern = re.compile(r'(?i)(author|by)\s*[:\-\s]*([A-Za-z,. ]+)')
title_pattern = re.compile(r'(?i)(publication title|title):?\s*[:\-\s]*(.*?)(?=\n|$)')
date_pattern = re.compile(r'(?i)(date|published):?\s*[:\-\s]*(\d{4}-\d{2}-\d{2})')

def clean_text(text):
    text = re.sub(r'http[s]?://\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_metadata(text):
    metadata = {}
    if (match := author_pattern.search(text)):
        metadata['Author'] = match.group(2).strip()
    if (match := title_pattern.search(text)):
        metadata['Publication Title'] = match.group(2).strip()
    if (match := date_pattern.search(text)):
        metadata['Date'] = match.group(2).strip()
    return metadata

def extract_titles_from_toc(file_path):
    logging.info("Extracting titles from Table of Contents.")
    with open(file_path, 'r', encoding='utf-8') as file:
        return re.findall(r'^\d+\.\s+(.*)', file.read(), re.MULTILINE)

def extract_text_beneath_duplicates(file_path, titles):
    logging.info(f"Extracting text beneath duplicates in {file_path}.")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    extracted_texts = []
    for title in titles:
        title_pattern = re.compile(r'(^|\n)' + re.escape(title) + r'(\n)', re.IGNORECASE)
        matches = title_pattern.finditer(content)
        
        for match in matches:
            start_pos = match.end()
            next_title_pos = next((m.start() for m in title_pattern.finditer(content[start_pos:])), None)
            extracted_text = content[start_pos:start_pos + next_title_pos].strip() if next_title_pos else content[start_pos:].strip()
            
            cleaned_text = clean_text(extracted_text)
            metadata = extract_metadata(extracted_text)
            extracted_texts.append((title, cleaned_text, metadata))

    logging.info(f"Extracted {len(extracted_texts)} text sections.")
    return extracted_texts

def summarize_text(text, summarizer):
    text = text.strip()
    if len(text) < 50:
        return "Text too short to summarize"

    max_chunk_size = 2048
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    logging.info("Summarizing text.")
    summaries = [summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
    logging.info("Summarization completed.")
    return " ".join(summaries)

def summarize_climate_insurance_correlation(text, summarizer):
    sentences = re.split(r'[.!?]', text)
    correlation_sentences = [s.strip() for s in sentences if 'climate change' in s.lower() and 'insurance' in s.lower()]
    return summarize_text(" ".join(correlation_sentences), summarizer) if correlation_sentences else "No correlation between climate change and insurance found."

def process_text_item_in_threads(text_data, summarizer):
    title, text, metadata = text_data
    summary = summarize_text(text, summarizer)
    correlation = summarize_climate_insurance_correlation(text, summarizer)
    return title, text, summary, correlation, metadata

def save_to_csv_incremental(data, output_file):
    df = pd.DataFrame([data], columns=[
        "Title", "Extracted Text", "Summary", "Climate-Insurance Correlation Summary",
        "Author", "Publication Title", "Date"
    ])
    df.to_csv(output_file, mode='a', header=False, index=False)
    logging.info("Row appended to CSV.")

def initialize_csv(output_file):
    df = pd.DataFrame(columns=[
        "Title", "Extracted Text", "Summary", "Climate-Insurance Correlation Summary",
        "Author", "Publication Title", "Date"
    ])
    df.to_csv(output_file, index=False)
    logging.info(f"CSV initialized at {output_file}")

def parallel_process_texts_in_threads(extracted_texts, summarizer, output_file):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_text = {executor.submit(process_text_item_in_threads, text, summarizer): text for text in extracted_texts}
        
        try:
            for future in as_completed(future_to_text, timeout=TIMEOUT):
                try:
                    title, text, summary, correlation, metadata = future.result()
                    row_data = (
                        title, text, summary, correlation, 
                        metadata.get("Author", "N/A"), 
                        metadata.get("Publication Title", "N/A"), 
                        metadata.get("Date", "N/A")
                    )
                    save_to_csv_incremental(row_data, output_file)
                except Exception as e:
                    logging.warning(f"Error processing text: {e}")
                    
        except FuturesTimeoutError:
            logging.warning("Time limit exceeded. Saving available results.")

def main(input_folder, output_file):
    start_time = time.time()
    
    initialize_csv(output_file)
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    # Loop through each .txt file in the specified folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            logging.info(f"Processing file: {file_path}")
            
            # Extract titles and text from each file
            titles = extract_titles_from_toc(file_path)
            extracted_texts = extract_text_beneath_duplicates(file_path, titles)
            
            # Process texts and write results to CSV incrementally
            parallel_process_texts_in_threads(extracted_texts, summarizer, output_file)

    logging.info("All files processed successfully.")

# Define folder and output path
input_folder = 'proquest_docs'
output_file = 'output_data.csv'

# Run the main function
main(input_folder, output_file)
