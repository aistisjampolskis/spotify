import numpy as np
import pandas as pd



pd.options.display.max_columns = None
pd.options.display.max_rows = None
df = pd.read_csv("spotifytoptracks.csv", header=0)
number_of_observations = df.shape[0]
print("Number of observations", number_of_observations)
number_of_features = df.shape[1]
print("Number of features:", number_of_features)

# Get a list of all the column names
column_names = list(df.columns)

# Initialize a list to store the categorical features
categorical_features = []

# Iterate through each column name
for column_name in column_names:
    # Check the data type of the column
    column_dtype = df[column_name].dtype
    # If the data type is object, add the column name to the list of categorical features
    if column_dtype == 'object':
        categorical_features.append(column_name)

# Print the list of categorical features
print("Categorial features", categorical_features)

# Get the data types of each column
dtypes = df.dtypes

# Filter the columns by data type to get only the numeric ones
numeric_cols = dtypes[dtypes != 'object'].index

# Print the numeric column names
print("Numeric features", numeric_cols)

# Group the data by the "artist" column
grouped = df.groupby('artist')

# Find the duplicates by filtering groups with more than 1 row
duplicates = grouped.filter(lambda x: len(x) > 1)

# Group the duplicates by the "artist" column again
grouped_duplicates = duplicates.groupby('artist')

# Count the number of duplicates for each artist
duplicate_counts = grouped_duplicates.count()['track_name']

# Print the result
print("Dublicates", duplicate_counts)

# count the number of times each artist appears
artist_counts = df["artist"].value_counts()

# Find the maximum count
max_count = max(artist_counts)

# Filter the artist_counts to include only the artists with the maximum count
max_artists = artist_counts[artist_counts == max_count].index.tolist()

# Print the artists and their counts
for artist in max_artists:
    count = artist_counts[artist]

# Find the most popular artist(s)
most_popular_artists = max_artists
print("The most popular artist(s) is/are:", most_popular_artists)

# Count the number of artists in total
num_artists = df["artist"].nunique()
print("Number of artists in the top 50 in total:", num_artists)

# Group the data by album and count the number of unique track names in each group
album_counts = df.groupby('album')['track_name'].nunique()

# Filter the groups where the number of unique track names is greater than 1
multi_track_albums = album_counts[album_counts > 1]

if len(multi_track_albums) > 0:
    print("There are {} albums that have more than 1 popular track:".format(len(multi_track_albums)))
    print(multi_track_albums)
else:
    print("There are no albums that have more than 1 popular track.")

# Count the number of albums in total
num_albums = df['album'].nunique()
print("The number of albums in total with songs in the top 50 is:", num_albums)

# Filter the DataFrame to only include tracks with danceability score above 0.7
danceable_tracks = df[df['danceability'] > 0.7]

# Print the names of the danceable tracks
print("The tracks with a danceability score above 0.7 are:")
print(danceable_tracks['track_name'])

# Filter the dataframe to include only the tracks with danceability below 0.4
filtered_df = df[df['danceability'] < 0.4]

# Get the track names of the filtered tracks
track_names = filtered_df['track_name'].tolist()

# Print the track names
print("Tracks with danceability below 0.4:")
for track_name in track_names:
    print(track_name)

# Filter the tracks by loudness
loud_tracks = df[df['loudness'] > -5]

# Select the track names from the filtered dataframe
loud_track_names = loud_tracks['track_name']

# Print the results
print("Tracks with loudness above -5:")
print(loud_track_names)

# Filter the dataframe to include only the tracks with loudness below -8
filtered_df = df[df['loudness'] < -8]

# Print the names of the tracks that satisfy the condition
print("Tracks with loudness below -8:")
for track_name in filtered_df['track_name']:
    print(track_name)

# Sort the DataFrame by track duration
sorted_by_duration = df.sort_values('duration_ms', ascending=False)

# Get the name of the longest track
longest_track = sorted_by_duration.iloc[0]['track_name']

# Print the result
print("The longest track is:", longest_track)

# Sort the dataframe by duration_ms
df_sorted = df.sort_values(by=['duration_ms'])

# Get the name of the shortest track (the first row after sorting)
shortest_track = df_sorted.iloc[0]['track_name']

print("The shortest track is:", shortest_track)

# Group the data by genre and count the number of occurrences of each genre
genre_counts = df['genre'].value_counts()

# Find the genre with the maximum count
most_popular_genre = genre_counts.idxmax()

# Print the result
print("The most popular genre is:", most_popular_genre)

# Group the data by genre and count the number of tracks in each group
genre_counts = df.groupby('genre')['track_name'].count()

# Filter the groups with only one track
genres_with_one_song = genre_counts[genre_counts == 1].index

# Print the names of the genres with only one song
print("Genres with just one song on the top 50:")
for genre in genres_with_one_song:
    print(genre)

num_genres = df['genre'].nunique()
print("Number of genres in total:", num_genres)
#----------------------------------------------------------
corr_matrix = df.corr(numeric_only=True)
strong_corr = corr_matrix[corr_matrix > 0.7]
strong_corr = strong_corr[strong_corr < 1.0].dropna(how='all', axis=1).dropna(how='all', axis=0)

print("Strongly positively correlated features:")
print(strong_corr.fillna(0))


# Find strongly negatively correlated features
strongly_neg_corr = (corr_matrix < -0.7) & (corr_matrix != -1)

# Print the result
if strongly_neg_corr.any().any():
    print("Strongly negatively correlated features:")
    print(strongly_neg_corr.stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'feature 1', 'level_1': 'feature 2'}))
else:
    print("No strongly negatively correlated features found.")
#--------------------------
# calculate the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# create a mask to ignore the diagonal values
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# create a new dataframe with the absolute correlation values
corr_abs = corr_matrix.abs().mask(mask)

# find the features that have an absolute correlation less than 0.1
uncorrelated_features = corr_abs[corr_abs < 0.1].dropna(axis=1, how='all').columns.tolist()

# print the uncorrelated features
print("Uncorrelated features:")
print(uncorrelated_features)
#----------------------
# Group the data by genre and calculate the mean danceability score for each genre
danceability_by_genre = df.groupby('genre')['danceability'].mean().sort_values(ascending=False)

# Print the results
print("Danceability by genre:" ,danceability_by_genre)
#-------------------------
genres = ['Pop', 'Hip-Hop/Rap', 'Dance/Electronic', 'Alternative/Indie']

loudness_by_genre = df.loc[df['genre'].isin(genres)].groupby('genre')['loudness'].mean().sort_values(ascending=False)

print("Loudness by genre (descending):", loudness_by_genre)
#-------------------
# Filter the dataset to include only the relevant genres
genres = ["Pop", "Hip-Hop/Rap", "Dance/Electronic", "Alternative/Indie"]
df = df[df["genre"].isin(genres)]

# Group the data by genre and calculate the mean acousticness score for each genre
acousticness_by_genre = df.groupby("genre")["acousticness"].mean().sort_values(ascending=False)

# Print the results
print("Acousticness by genre:", acousticness_by_genre)