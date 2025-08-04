from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
from transformers import pipeline
import os

# Initialize the app
app = FastAPI(title="Text Summarization API", version="1.0.0")

# Add CORS middleware for web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load summarizer once (will be cached)
try:
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to a more reliable model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class TextInput(BaseModel):
    text: str

def chunk_text(text, max_words=600):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_length + word_count > max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = word_count
        else:
            current_chunk.append(sentence)
            current_length += word_count
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def truncate_to_words(text, max_words=30):
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]).rstrip('.,;:') + '.'
    return text

def scale_summary_length(word_count):
    """
    Given input word count, return (min_length, max_length)
    for summary with sensible scaling
    """
    if word_count < 50:
        return (10, 20)
    elif word_count < 150:
        return (20, 40)
    elif word_count < 500:
        return (40, 80)
    else:
        return (50, 150)

def summarize_text(text):
    word_count = len(text.split())
    min_len, max_len = scale_summary_length(word_count)

    if word_count <= 500:
        # Direct summarization for smaller texts
        summary = summarizer(text,
                             max_length=max_len,
                             min_length=min_len,
                             do_sample=False)
        return summary[0]['summary_text']
    else:
        # Long text: hierarchical summarization with scaled lengths
        chunks = chunk_text(text, max_words=600)
        # For first pass on chunks, use shorter summaries
        first_pass_min = max(15, min_len // 3)
        first_pass_max = max(30, max_len // 3)

        chunk_summaries = []
        for chunk in chunks:
            summary = summarizer(chunk,
                                 max_length=first_pass_max,
                                 min_length=first_pass_min,
                                 do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])

        combined_summary = " ".join(chunk_summaries)
        final_summary = summarizer(combined_summary,
                                  max_length=max_len,
                                  min_length=min_len,
                                  do_sample=False)[0]['summary_text']
        return final_summary

def improve_summary(summary, max_words=30):
    # Remove awkward or repetitive phrases
    summary = re.sub(r'(\bthis is a good example of\b.*?\. )', '', summary, flags=re.I)
    
    # Fix repeated words
    summary = re.sub(r'\b(\w+)( \1\b)+', r'\1', summary)

    # Remove duplicated industry mentions
    summary = re.sub(r'(\b\w+\b)( and \1\b)+', r'\1', summary, flags=re.I)

    # Fix spacing before punctuation
    summary = re.sub(r'\s+([.,])', r'\1', summary)

    # Merge sentences with conjunctions smoothly
    summary = re.sub(r'\. ([Tt]he)', r'; \1', summary)

    # Remove excessive whitespace
    summary = re.sub(r'\s{2,}', ' ', summary)
    summary = summary.strip()

    # Capitalize first letter
    if summary:
        summary = summary[0].upper() + summary[1:]

    # Ensure ending with period
    if summary and not summary.endswith('.'):
        summary += '.'

    # Truncate to max_words
    max_summary_words = max(20, min(50, max_words))
    summary = truncate_to_words(summary, max_words=max_summary_words)

    return summary.strip()

@app.get("/")
def read_root():
    return {
        "message": "Text Summarization API is running!",
        "version": "1.0.0",
        "endpoints": {
            "POST /summarize": "Summarize text",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": summarizer is not None}

@app.post("/summarize")
def summarize_endpoint(input_data: TextInput):
    try:
        if not input_data.text or len(input_data.text.strip()) < 10:
            return {"error": "Text too short. Please provide at least 10 characters."}
        
        raw_summary = summarize_text(input_data.text)
        max_words = min(50, max(20, len(input_data.text.split()) // 10))
        final_summary = improve_summary(raw_summary, max_words=max_words)
        
        return {
            "summary": final_summary,
            "original_length": len(input_data.text.split()),
            "summary_length": len(final_summary.split()),
            "compression_ratio": f"{len(final_summary.split()) / len(input_data.text.split()) * 100:.1f}%"
        }
    except Exception as e:
        return {"error": f"Summarization failed: {str(e)}"}

# For Vercel
app_handler = app
