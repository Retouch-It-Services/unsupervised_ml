📊 Analysis of TEDx Talks Dataset

1. Data Loading and Exploration
1.1 Dataset Overview

Total Records: 4,467

Columns:

idx: Unique identifier

main_speaker: Speaker name

title: Talk title

details: Talk description

posted: Posting date

url: Talk URL

num_views: Number of views (only 209 non-null values)

1.2 Initial Insights

Missing Data: num_views has ~95% missing values (4,258 nulls).

Temporal Coverage: Talks range from 2006–2020, with 2020 being the most active year.

Speaker Data: Only 1 missing value in main_speaker.

2. Data Preprocessing
2.1 Handling Missing Values

Dropped rows with missing main_speaker.

Retained num_views despite missing values (may still be useful for partial analysis).

2.2 Feature Engineering

Extracted year and month from posted.

Combined title + details into a unified text column for NLP tasks.

2.3 Text Preprocessing

Stopword Removal (NLTK English stopwords).

Punctuation Removal.

Lowercasing for normalization.

3. Text Analysis
3.1 Word Frequency

Generated WordClouds for dominant terms.

Common themes observed:

Innovation & Technology

Social Issues & Change

Psychology & Self-Development

Science & Research

3.2 Temporal Trends

Talks increased steadily over the years.

Technology-related talks surged in recent years.

4. Unsupervised Learning
4.1 Feature Extraction

Used TF-IDF Vectorization to create a sparse document-term matrix.

4.2 Clustering

K-Means Clustering: Optimal k chosen via elbow method.

Hierarchical Clustering: Dendrogram used to visualize nested relationships.

4.3 Topic Modeling

Applied Latent Dirichlet Allocation (LDA) to uncover hidden topics.

Each document assigned probability distribution across latent themes.

5. Results
5.1 Cluster Insights

Technology & Innovation → AI, blockchain, future trends.

Social Issues → Equality, justice, activism.

Personal Development → Motivation, psychology, well-being.

Science & Education → Discoveries, teaching methods.

5.2 Topic Modeling Outcomes

Technology & Future Trends

Health & Medicine

Environment & Climate

Business & Entrepreneurship

Arts & Creativity

5.3 Cosine Similarity

Computed pairwise cosine similarity to measure talk similarity.

Built a recommendation framework: suggest talks based on semantic similarity.

6. Limitations

Missing Data: num_views too sparse for robust popularity analysis.

Text Preprocessing Loss: Contextual nuance (e.g., sarcasm, idioms) lost.

Subjectivity: Manual labeling of clusters/topics involves interpretation.

Scope: Only TEDx talks included → not fully representative of TED.

7. Technical Implementation

Libraries Used:

pandas, numpy → Data handling

matplotlib, seaborn → Visualization

nltk → Text preprocessing

scikit-learn → TF-IDF, clustering, similarity

wordcloud → Word frequency visualization

gensim / sklearn → LDA topic modeling