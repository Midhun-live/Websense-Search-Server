import os
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx
from typing import List, Set
import re
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from bs4 import BeautifulSoup, Tag

# Disable cuFFT plugin registration to avoid the error
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Enable memory growth for GPU (if you're using a GPU) to avoid memory allocation conflicts
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

app = FastAPI()

port = int(os.getenv("PORT", 8000))

class SearchRequest(BaseModel):
    url: HttpUrl
    query: str

class SearchResult(BaseModel):
    title: str
    content: str
    html: str
    path: str
    matchPercentage: float

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Initialize FAISS index
dimension = 768  # Dimension of the BERT embeddings
index = faiss.IndexFlatL2(dimension)

def calculate_match_percentage(text: str, query: str) -> float:
    text_lower = text.lower()
    query_lower = query.lower()
    match_count = text_lower.count(query_lower)
    total_words = len(text_lower.split())
    query_words = len(query_lower.split())
    return min((match_count * query_words / total_words) * 100, 100)

def tokenize_and_vectorize(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=500)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_specific_matches(soup: BeautifulSoup, query: str) -> List[tuple[str, str, float]]:
    matches = []
    seen_content: Set[str] = set()
    matched_elements: Set[Tag] = set()

    def process_element(element: Tag) -> bool:
        nonlocal matches, seen_content, matched_elements

        child_matched = False
        for child in element.children:
            if isinstance(child, Tag):
                if process_element(child):
                    child_matched = True
                    
        if child_matched:
            matched_elements.add(element)
            return True

        direct_text = ''.join(child for child in element.strings if isinstance(child, str)).strip()

        if direct_text and direct_text.lower() not in seen_content:
            text_lower = direct_text.lower()
            if query.lower() in text_lower:
                match_percentage = calculate_match_percentage(direct_text, query)
                
                element_html = str(element)
                matches.append((direct_text, element_html, match_percentage))
                seen_content.add(text_lower)
                matched_elements.add(element)
                return True

        return False

    elements = soup.find_all(['p', 'div', 'section', 'article', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    for element in elements:
        if element not in matched_elements:
            process_element(element)

    return matches

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.url))
            response.raise_for_status()
            content = response.text

        soup = BeautifulSoup(content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        matches = find_specific_matches(soup, request.query)
        
        if not matches:
            return []
        
        results = []
        seen_content = set()
        index.reset()  # Reset the index for each new search

        for idx, (text, html, match_percentage) in enumerate(matches):
            normalized_text = ' '.join(text.lower().split())
            
            if normalized_text not in seen_content and match_percentage >= 1:
                vector = tokenize_and_vectorize(text)
                index.add(vector)
                results.append(SearchResult(
                    title=f"Exact Match {idx + 1}",
                    content=text,
                    html=html,
                    path=str(request.url),
                    matchPercentage=round(match_percentage, 1)
                ))
                seen_content.add(normalized_text)

        query_vector = tokenize_and_vectorize(request.query)
        num_results = min(10, len(results))
        distances, indices = index.search(query_vector, num_results)

        ranked_results = [results[i] for i in indices[0] if i < len(results)]

        return ranked_results

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)