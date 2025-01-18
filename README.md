
# Web Search with FastAPI

This project provides a web search service that allows users to search for a specific query within the content of a webpage. The application processes the page, extracts relevant content, and ranks the matches based on similarity and match percentage.

# Features

 - Fetches the content of a URL and parses it.
 - Supports searching for specific text or queries within a webpage.
 - Uses BERT-based embeddings (DistilBERT) to compute semantic   
 - similarity between the query and webpage content.
 - Results are ranked using FAISS for fast vector search.
 - Returns search results with match percentage and HTML context.
 - Includes health check and search endpoints.

 # Requirements

 - Python 3.8+
 - FastAPI
 - pydantic
 - transformers (for BERT-based model)
 - torch (for PyTorch support)
 - faiss (for fast vector search)
 - httpx (for HTTP requests)
 - beautifulsoup4 (for HTML parsing)
 
 You can install the required dependencies using:
```bash
  pip install -r requirements.txt
```
# Setup

1. Clone this repository:
```bash
  git clone https://github.com/Midhun-live/Websense-Search-Server.git
  cd Websense-Search-Server
```
2. Install the dependencies:
```bash
  pip install -r requirements.txt
```
3. Make sure to set the PORT environment variable, or the default will be 8000:
```bash
  export PORT=8000  # or any other port number
```
4. Run the application:
```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
```

# API Endpoints
 
 1. Health Check

 - GET /health

 - Returns the health status of the service.

 Response:
 ```bash
  {
    "status": "ok"
  }
```

 2. Search

 - POST /search

 - Searches a webpage for a given query and returns relevant results.

 Request Body::
 ```bash
  {
    "url": "https://example.com",
    "query": "your search query"
  }
```

# Response Body:

Returns a list of SearchResult items. 

Each result contains:

-  title: The title or identifier of the match.
 - content: The matched content from the page.
 - html: The full HTML snippet where the  match was found.
 - path: The URL of the page.
 - matchPercentage: The match percentage between the query and the matched content.

# Error Handling
 The API returns appropriate HTTP error codes if something goes wrong:

 - 400: Bad Request (e.g., Invalid URL or query).
 - 500: Internal Server Error (e.g., failed to process the request).

# Configuration for FAISS Vector Database

1. **Vector Dimensions:** Ensure that the FAISS index dimension matches the embedding size of your model. For DistilBERT, it is 768.
2. **Initialization:** The FAISS index is initialized in the backend code. No additional setup is required unless you use a pre-trained FAISS index.
3. **Reset:** The index is reset for each new search request to avoid overlapping results.
