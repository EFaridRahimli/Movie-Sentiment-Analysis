# Movie Review Sentiment Analysis

An NLP pipeline that labels movie reviews as **positive or negative**, so a new streaming platform can power "Popular" / "Top" sections and recommendations without manually reading reviews.

## Results
| Metric (8,000 held-out reviews) | Score |
|---|---|
| Accuracy | **88.5%** |
| F1 (negative / positive) | 0.89 / 0.88 |

## Approach
![High-level design](hld.JPG)

1. **Data:** 40,000 labelled IMDB reviews ([Kaggle](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis)), perfectly balanced classes.
2. **Preprocessing:** HTML removal (BeautifulSoup), regex cleanup, tokenisation, stop-word removal, lemmatisation and stemming (NLTK).
3. **Features:** TF-IDF vectors plus a scaled **VADER** sentiment-intensity score.
4. **Model:** Multinomial Naive Bayes tuned with `GridSearchCV`.

## Limitations and next steps
- No neutral class, and a simple model struggles with complex language.
- Next: multi-class sentiment, deep-learning models, multilingual support and real-time labelling of new reviews.

## Tech stack
Python · NLTK · scikit-learn · BeautifulSoup · pandas · Seaborn

## Run it
Download `movie.csv` from the Kaggle link above, place it next to the notebook, then:
```bash
pip install pandas nltk scikit-learn beautifulsoup4 matplotlib seaborn jupyter
jupyter notebook sentiment_analysis.ipynb
```
