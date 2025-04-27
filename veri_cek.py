import praw
import pandas as pd

# Reddit API bilgileri ve kullanıcı giriş bilgileri
reddit = praw.Reddit(
    client_id="2S3hOWqblzRNHydtambiUw",
    client_secret="0Cnr1z5U9G2uLOcJjkhxoGlp2QI49Q",
    username="yapmacj",
    password="Yunusemre2001",
    user_agent="dil_isleme_project/0.1 by yapmacj"
)

subreddit = reddit.subreddit("soccer")  # r/soccer subreddit'inden çekiyoruz

comments = []

# 200 gönderiyi tarayarak yorumları çekiyoruz
for submission in subreddit.hot(limit=200):
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        if comment.body and len(comment.body) > 10:  # boş ve kısa yorumları alma
            comments.append(comment.body)

# DataFrame'e kaydedelim
df = pd.DataFrame(comments, columns=["comment"])

# CSV dosyasına yazalım
df.to_csv("reddit_soccer_comments.csv", index=False, encoding='utf-8')

print(f"✅ Toplam {len(comments)} yorum çekildi ve reddit_soccer_comments.csv dosyasına kaydedildi!")
