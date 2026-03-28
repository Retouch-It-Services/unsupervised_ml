# Report on TEDx Talks Using Unsupervised Learning

## 1. Overview
This work investigates TEDx talks with the help of unsupervised machine learning. The dataset includes speaker information, talk titles, descriptions, publication dates, URLs, and view counts. The primary objective is to clean and process the text data, then group talks by similarity in content.

## 2. Dataset Exploration
### 2.1 General Structure
The dataset consists of 4,467 rows with these attributes:
- **idx**: Unique ID for each record  
- **main_speaker**: Speaker’s name  
- **title**: Talk title  
- **details**: Description of the talk  
- **posted**: Date posted  
- **url**: Link to the talk  
- **num_views**: View counts (only 209 valid entries)  

### 2.2 Observations
- The `num_views` column is heavily incomplete (over 4,200 missing).  
- Most entries are recent, especially from 2020.  
- Coverage spans 2006–2020.  

## 3. Data Preparation
### 3.1 Missing Values
- Dropped one row with a missing `main_speaker`.  
- Retained other incomplete fields, especially `num_views`.  

### 3.2 Feature Engineering
- Extracted year and month from `posted`.  
- Merged `title` and `details` into one text field.  

### 3.3 Text Cleaning
- Removed stopwords with NLTK.  
- Stripped punctuation.  
- Converted all text to lowercase.  

## 4. Text Insights
### 4.1 Word Frequency
Word clouds highlighted recurring themes such as:
- Technology and innovation  
- Society and change  
- Psychology and self-growth  
- Science and discovery  

### 4.2 Time Trends
- Talks have grown in number steadily over the years.  
- Technology-related topics rose sharply in recent years.  

## 5. Unsupervised Learning Process
### 5.1 Feature Representation
TF-IDF was applied to represent text numerically as a document-term matrix.  

### 5.2 Clustering
- **K-Means**: Optimal clusters chosen with elbow method. Clear thematic groups identified.  
- **Hierarchical Clustering**: Dendrogram built to reveal nested groupings.  

### 5.3 Topic Modeling
Latent Dirichlet Allocation (LDA) was used to uncover hidden topics and distribute them across talks.  

## 6. Findings
### 6.1 Clusters
Distinct groups observed:
- Technology and innovation (AI, blockchain, etc.)  
- Social issues (justice, inequality)  
- Self-development and psychology  
- Education and science  

### 6.2 Topics from LDA
Key recurring themes included:
- Future technology  
- Medicine and health  
- Environmental challenges  
- Entrepreneurship and business  
- Creative arts  

### 6.3 Similarity Analysis
Cosine similarity scores helped:
- Detect related talks  
- Suggest recommendations for topic-specific interests  

## 7. Constraints
- **Incomplete Data**: View counts missing for most talks.  
- **Preprocessing Limits**: Some contextual meaning lost.  
- **Interpretation Bias**: Human judgment required for topic labeling.  
- **Scope**: Only TEDx, not all TED events.  

## 8. Tools and Workflow
**Libraries**: pandas, numpy, matplotlib, nltk, sklearn, wordcloud.  

**Steps**:  
1. Load and clean data  
2. Process text  
3. Extract features (TF-IDF)  
4. Reduce dimensions (PCA/t-SNE)  
5. Cluster (K-Means, Hierarchical)  
6. Apply LDA  
7. Visualize and interpret results  
