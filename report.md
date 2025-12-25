 
# TEDx Talks Recommendation System Report

## Overview
This project builds an unsupervised machine learning recommendation system for TEDx talks using text data. The workflow includes data preprocessing, visualization, and similarity-based recommendations.

## Data Preprocessing
- Loaded TEDx talks data from `tedx_datase.csv`.
- Combined and cleaned the `title` and `details` columns.
- Removed stopwords and punctuation from the text data.

## Visualization
- Generated a word cloud to visualize the most frequent words in the talks' details.

## Recommendation System
- Used TF-IDF vectorization to represent talk details numerically.
- Calculated cosine similarity and Pearson correlation between talks.
- Implemented a function to recommend similar talks based on user input.

## Results
- The system can recommend talks similar to a given topic or description using unsupervised learning techniques.

## Libraries Used
- pandas, numpy, matplotlib, nltk, wordcloud, scikit-learn, scipy

## Conclusion
The notebook demonstrates a complete pipeline for building a content-based recommendation system for TEDx talks using unsupervised