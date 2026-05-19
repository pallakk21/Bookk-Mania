import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

print("Libraries loaded!")

books = pd.read_csv('Books.csv', low_memory=False)
users = pd.read_csv('Users.csv', low_memory=False)
ratings = pd.read_csv('Ratings.csv', low_memory=False)
print("CSVs loaded!")

ratings_with_name = ratings.merge(books, on='ISBN')
print("Merge done! Rows:", len(ratings_with_name))

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]
print("Popular df ready! Shape:", popular_df.shape)

x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index
print("Active users:", len(padhe_likhe_users))

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]
print("Filtered ratings:", len(filtered_rating))

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
print("Famous books:", len(famous_books))

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
print("Final ratings shape:", final_ratings.shape)

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
print("Pivot table ready! Shape:", pt.shape)

similarity_scores = cosine_similarity(pt)
print("Similarity scores ready!")

pickle.dump(popular_df, open('popular.pkl', 'wb'), protocol=4)
pickle.dump(pt, open('pt.pkl', 'wb'), protocol=4)
pickle.dump(books, open('books.pkl', 'wb'), protocol=4)
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'), protocol=4)

print("Done! Sab pkl files ready!")