 Unsupervised Learning Analysis of TEDx Talks
Abstract

This study analyzes 4,467 TEDx talk records using unsupervised learning techniques to uncover thematic clusters, temporal trends, and semantic relationships among talks. Data preprocessing included handling missing values, feature engineering, and text cleaning. Key methods applied were TF-IDF vectorization, K-Means clustering, hierarchical clustering, and Latent Dirichlet Allocation (LDA). Results reveal distinct clusters in technology, social issues, personal development, and science, while limitations include missing engagement data and contextual loss during text preprocessing.

1. Introduction

TEDx talks provide a valuable dataset for exploring global themes in education, technology, social issues, and personal growth. This work applies natural language processing (NLP) and machine learning to:

Identify hidden themes and clusters in TEDx talks.

Analyze temporal trends across years.

Generate insights for recommendation systems based on similarity analysis.

2. Data Description

Dataset Size: 4,467 records

Columns:

idx – Identifier

main_speaker – Speaker name

title – Talk title

details – Talk description

posted – Publication date

url – Talk link

num_views – View count (sparse; only 209 non-null values)

Observations:

Missing values are significant in num_views.

Posting frequency increases from 2006 → 2020, peaking in 2020.

Only one missing value in main_speaker.

3. Methodology
3.1 Preprocessing

Removed rows with missing speaker names.

Extracted year and month from posting date.

Created a combined text column (title + details).

Text normalized via:

Stopword removal

Punctuation removal

Case normalization

3.2 Feature Extraction

Used TF-IDF to transform text into a numerical matrix.

Applied dimensionality reduction (PCA/t-SNE) for visualization.

3.3 Algorithms Applied

K-Means Clustering – Optimal k chosen with elbow method.

Hierarchical Clustering – Produced dendrograms for hierarchical relationships.

LDA Topic Modeling – Extracted latent topics.

Cosine Similarity – Measured semantic similarity between talks.

4. Results
4.1 Clustering Outcomes

Cluster 1: Technology & Innovation – AI, blockchain, digital future.

Cluster 2: Social Issues – Inequality, justice, activism.

Cluster 3: Personal Development – Psychology, motivation, well-being.

Cluster 4: Science & Education – Research, discoveries, learning.

4.2 Topic Modeling

Key latent themes identified:

Technology & Future Trends

Health & Medicine

Environment & Climate Change

Business & Entrepreneurship

Arts & Creativity

4.3 Similarity Analysis

Talks grouped by cosine similarity allow content-based recommendations.

Example: A talk on AI is likely to be linked with blockchain or automation talks.

5. Discussion

The analysis highlights how TEDx talks reflect global intellectual and social trends. Technology dominates in recent years, while social issues remain a consistent theme. Unsupervised clustering provides valuable insights for recommendation systems, but interpretability depends on analyst judgment.

6. Limitations

Sparse Engagement Data: num_views too incomplete for popularity modeling.

Loss of Context: Preprocessing removes linguistic nuance.

Scope Restriction: Focus limited to TEDx talks dataset.

7. Conclusion

Unsupervised learning techniques successfully uncovered meaningful clusters and topics within TEDx talks. These findings demonstrate the potential of NLP-driven methods for analyzing large collections of educational and social discourse.

8. Tools and Libraries

pandas, numpy – Data manipulation

nltk – Text preprocessing

scikit-learn – Clustering, similarity, TF-IDF

gensim/sklearn – Topic modeling

matplotlib, seaborn, wordcloud – Visualization