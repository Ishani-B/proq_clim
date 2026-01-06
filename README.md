# ProQuest Text Extractor + Summarizer (Climate–Insurance Focus)

A Python pipeline that processes `.txt` documents (e.g., exported research articles), extracts sections based on Table of Contents-style headings, summarizes each section using a Transformer model, and writes structured outputs to a CSV. It also produces a targeted mini-summary for sentences that mention both **climate change** and **insurance**.

## What it does

For every `.txt` file in an input folder:

1. Extracts section titles from a Table of Contents pattern (`1. Title`, `2. Title`, ...)
2. Finds matching occurrences of those titles in the document and extracts the text beneath them
3. Cleans extracted text (removes URLs, normalizes whitespace)
4. Extracts basic metadata when present:
   - Author
   - Publication Title
   - Date (`YYYY-MM-DD`)
5. Summarizes each extracted section using `facebook/bart-large-cnn`
6. Generates a focused “Climate–Insurance Correlation” summary by filtering to sentences containing both keywords
7. Appends results incrementally to `output_data.csv` (safe for long runs)

## Output

Creates/overwrites a CSV with the following columns:

- `Title`
- `Extracted Text`
- `Summary`
- `Climate-Insurance Correlation Summary`
- `Author`
- `Publication Title`
- `Date`

## Tech stack

- Python 3.9+
- `transformers` (Hugging Face) summarization pipeline
- `facebook/bart-large-cnn`
- `pandas`
- `concurrent.futures` for multithreading

## My Purpose

I developed this program while conducting independent research on the relationship between the finance and insurance industries and the evolution of catastrophic risk modeling, particularly following major climate events such as **Hurricane Andrew** and **Hurricane Katrina**.

My goal was to create a scalable tool that could process large volumes of academic and industry publications, extract relevant sections automatically, and generate structured summaries that make it easier to analyze how climate risk, insurance markets, and financial institutions influence one another over time.

Rather than manually reading hundreds of long-form documents, this pipeline allows me to:
- rapidly identify key themes and patterns across sources,
- isolate content that explicitly connects **climate change** and **insurance**, and
- produce a clean, analyzable dataset for downstream research and visualization.

This project sits at the intersection of **natural language processing**, **data engineering**, and **climate risk analysis**, and reflects my interest in building practical tools that support complex real-world research workflows.

