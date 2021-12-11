from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer as tfv
from sklearn.cluster import KMeans

df_train=pd.read_csv('train_nykaa_review_sentiment.csv',sep=',',encoding='utf8')
for i in df_train['content']:
    i = str(i)
#removing numbers and special characters
df_train.columns.str.replace('[#,@,&]', '')
df_train.columns.str.replace('[^A-Za-z\s]+', '')
df_train.columns.str.replace('\d+', '')
#print(df_train)

#wordcloud
wdc=df_train['content'].values  #created list from dataframe
stopwords=set(STOPWORDS)
stopwords.update(["Nykaa","nykaa","nykaa'","app'"])
wordcloud=WordCloud(stopwords=stopwords, background_color="white").generate(str(wdc))
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

#barplot
train_pos=df_train[df_train['sentiment_labels'] == 'POSITIVE']
train_neg=df_train[df_train['sentiment_labels'] == 'NEGATIVE']
train_neu=df_train[df_train['sentiment_labels'] == 'NEUTRAL']
lst = list()
lst.append(len(train_pos))
lst.append(len(train_neg))
lst.append(len(train_neu))
sentiment=['POSITIVE','NEGATIVE','NEUTRAL']
# plt.bar(sentiment, lst, color ='green',width = 0.4)
# plt.xlabel("Sentiment label")
# plt.ylabel("Frequency")
# plt.title("Bar plot for sentiment labels")
# plt.show()

#using only content column to identify the sentiments
# df_train_cont=df_train['content']
# print(df_train_cont)
# for i in df_train_cont:
#     sentiment_blob= TextBlob(i)
    #print('{:40} :   {: 01.2f}    :   {:01.2f}'.format(i[:40],sentiment_blob.polarity,sentiment_blob.subjectivity))

#categorize polarity into sentiment labels
df_train_cont=df_train['content']
values=[0,0,0]
for i in df_train_cont:
    i=str(i)
    sentiment_blob= TextBlob(i)
    polarity = round((sentiment_blob.polarity + 1) * 3) % 3
    values[polarity] = values[polarity] + 1

# print("Final summarized counts :", values)
colors=["Green","Blue","Red"]
# print("\n Pie Representation \n-------------------")
plt.pie(values, labels=sentiment, colors=colors,autopct='%1.1f%%', startangle=140)
plt.axis('equal')
# plt.show()

#vectorizing using TF-ID
vectorizer = tfv(stop_words='english')
hash_matrix=vectorizer.fit_transform(df_train_cont.values.astype('U'))
# print(hash_matrix)
# print("\n Feature names Identified :\n")
# print(vectorizer.get_feature_names())
# print(df_train_cont.isna())

#elbow method to find optimal cluster size
sosd = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(hash_matrix)
    sosd.append(km.inertia_)
print("Sum of squared distances : " ,sosd)
plt.plot(K, sosd, 'bx-')
plt.xlabel('Cluster count')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal Cluster Size')
plt.show()


#clustering using Kmeans
kmeans = KMeans(n_clusters=3).fit(hash_matrix)
clusters=kmeans.labels_
# for group in set(clusters):
#     print("\nGroup : ",group,"\n-------------------")

    # for i in df_train_cont.index:
    #     if (clusters[i] == group):
    #         print(df_train_cont[i])