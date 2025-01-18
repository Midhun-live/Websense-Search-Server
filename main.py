from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx
from typing import List, Set
import re
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

try:
    from bs4 import BeautifulSoup, Tag
except ImportError as e:
    print(f"Error: {e}. Please install all required packages.")
    print("Run: pip install fastapi uvicorn beautifulsoup4 httpx numpy transformers faiss-cpu")
    exit(1)

app = FastAPI()

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
    """Calculate the match percentage for a given text and query."""
    text_lower = text.lower()
    query_lower = query.lower()
    match_count = text_lower.count(query_lower)
    total_words = len(text_lower.split())
    query_words = len(query_lower.split())
    return min((match_count * query_words / total_words) * 100, 100)

def tokenize_and_vectorize(text: str):
    """Tokenize and vectorize the input text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=500)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_specific_matches(soup: BeautifulSoup, query: str) -> List[tuple[str, str, float]]:
    """Find exact matches in the HTML content and return only the specific matching elements."""
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

        
        if len(results) <= 10:
            query_vector = tokenize_and_vectorize(request.query)
            distances, indices = index.search(query_vector,  len(results)) 
            ranked_results = [results[i] for i in indices[0] if i < len(results)]

            return ranked_results 
       
        else : 

            query_vector = tokenize_and_vectorize(request.query)
            distances, indices = index.search(query_vector, 10)  


            ranked_results = [results[i] for i in indices[0] if i < len(results)]

            return ranked_results

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)