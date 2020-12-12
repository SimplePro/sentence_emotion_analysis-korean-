import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from eunjeon import Mecab
from sklearn.model_selection import train_test_split
import pandas as pd

mecab = Mecab()
tfv = TfidfVectorizer(tokenizer = mecab.morphs, ngram_range=(1, 2), min_df=3, max_df=0.9)
loaded_model = joblib.load("./analysis_model.pkl")

data = pd.read_csv("./datasets/sentences.csv", index_col=0)
data["sentence"] = data["sentence"].fillna("0")

features = data["sentence"]
label = data["emotion"]

train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.2, random_state=0)

tfv.fit(train_x)


def emo(s):
    sentence = tfv.transform([s])
    predict = loaded_model.predict(sentence)[0]
    return "긍정" if predict == 1 else "부정"