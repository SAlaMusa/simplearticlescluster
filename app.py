import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

NEWS_API_KEY = 'f0a92cc9066f46bab677a0216babdceb'  # Replace with your News API key

def get_news(api_key, query, num_articles=50):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize={num_articles}'
    response = requests.get(url)
    data = response.json()
    articles = [article['title'] + ' ' + article['description'] for article in data['articles']]
    return articles, [article['url'] for article in data['articles']]  # Return both articles and URLs

def cluster_news(articles):
    vectorizer = TfidfVectorizer(stop_words='english')
    article_matrix = vectorizer.fit_transform(articles)

    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(article_matrix)

    clustered_articles = {cluster: {'category': f'Category {cluster + 1}', 'articles': []} for cluster in range(max(clusters) + 1)}
    for i, cluster in enumerate(clusters):
        clustered_articles[cluster]['articles'].append(i)

    return clustered_articles

def main():
    st.set_page_config(page_title="Clustered News Articles", page_icon="📰")

    query = 'technology'
    num_articles = 50
    articles, news_links = get_news(NEWS_API_KEY, query, num_articles)
    clustered_articles = cluster_news(articles)

    st.markdown(
        """
        <style>
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
            }
            .grid-item {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px; /* Added margin-bottom */
            }
        </style>
        <div style="background-color:#1A202C; padding: 20px; margin-bottom: 20px;">
            <h1 style="color:#ffffff; font-size: 30px; font-weight: bold;">Clustered News Articles</h1>
            <p style="color:#D2D6DC; font-size: 18px;">This is a list of similar technology-related articles. To cluster articles and determine similarity, the Kmeans approach was used. Scraped from News Org</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for cluster, details in clustered_articles.items():
        st.markdown(f"## Cluster {cluster} - {details['category']}")
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)

        for article_index in details['articles']:
            st.markdown(f'<div class="grid-item"><a href="{news_links[article_index]}" target="_blank">{articles[article_index].splitlines()[0]}</a></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()

