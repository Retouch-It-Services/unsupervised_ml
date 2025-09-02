# TED Talk Recommendation System - Analysis

## Project Overview
This project builds a content-based recommendation system for TEDx talks using unsupervised machine learning techniques. The system analyzes talk content (titles and descriptions) to find similar talks based on textual similarity.

## Data Description
The dataset contains:
- 4,467 TEDx talks
- Columns: idx, main_speaker, title, details, posted, url, num_views
- Key text features: title and details (combined for analysis)
- Many talks have missing view counts (only 209 have num_views data)

## Methodology

### 1. Text Preprocessing
- Combined title and details into single text features
- Converted to lowercase
- Removed punctuation and stopwords
- Cleaned text for better feature extraction

### 2. Feature Engineering
- Used TF-IDF (Term Frequency-Inverse Document Frequency) for vectorization
- This technique weights words based on their importance in documents
- Creates a numerical representation of text content

### 3. Similarity Measurement
- Cosine similarity to measure similarity between talk vectors
- Range: 0 (completely dissimilar) to 1 (identical content)
- Efficient for high-dimensional sparse data like text

### 4. Recommendation Logic
- For each talk, find the most similar talks based on cosine similarity
- Exclude the query talk itself from recommendations
- Return top N most similar talks

## Key Findings

### Data Insights
- Text data is rich and descriptive, good for content-based recommendations
- Many talks have minimal metadata (missing views, sparse speaker info)
- Year distribution shows concentration in recent years

### Technical Implementation
- TF-IDF effectively captures content similarities
- Cosine similarity works well for text recommendation
- The system can handle the full dataset efficiently

## Limitations
1. Content-based only (no collaborative filtering)
2. Doesn't consider user preferences or viewing history
3. Limited by text quality and completeness
4. No popularity or recency factors incorporated

## Future Enhancements
1. Add collaborative filtering with user data
2. Incorporate view counts and popularity metrics
3. Use advanced embeddings (Word2Vec, BERT)
4. Add topic modeling for better categorization
5. Create a web interface for interactive recommendations