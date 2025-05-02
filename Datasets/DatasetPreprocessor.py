import pandas as pd
import numpy as np



def preprocess_shortjokes(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.rename(columns={'ID': 'ID', 'Joke': 'Body'})
    df['Title'] = pd.NA
    df['Category'] = pd.NA
    df['Rating'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False) # index=False nu adauga index la randuri


# preprocess_shortjokes('Original-Datasets/Positive-Examples/shortjokes/shortjokes.csv', 'Preprocessed-Datasets/Positive-Examples/shortjokes/shortjokes.csv')


def preprocess_reddit_jokes(input_path, output_path):
    df = pd.read_json(input_path)

    df = df.rename(columns={'id': 'ID', 'title': 'Title', 'body': 'Body', 'score': 'Rating'})
    df['Category'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_reddit_jokes('Original-Datasets/Positive-Examples/joke-dataset/reddit_jokes.json', 'Preprocessed-Datasets/Positive-Examples/joke-dataset/reddit_jokes.csv')


def preprocess_stupidstuff(input_path, output_path):
    df = pd.read_json(input_path)

    df = df.rename(columns={'id': 'ID', 'category': 'Category', 'body': 'Body', 'rating': 'Rating'})
    df['Title'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_stupidstuff('Original-Datasets/Positive-Examples/joke-dataset/stupidstuff.json', 'Preprocessed-Datasets/Positive-Examples/joke-dataset/stupidstuff.csv')


def preprocess_wocka(input_path, output_path):
    df = pd.read_json(input_path)

    df = df.rename(columns={'id': 'ID', 'title': 'Title', 'category': 'Category', 'body': 'Body'})
    df['Rating'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_wocka('Original-Datasets/Positive-Examples/joke-dataset/wocka.json', 'Preprocessed-Datasets/Positive-Examples/joke-dataset/wocka.csv')


def preprocess_jester(input_path, ratings_path, output_path):
    df = pd.read_csv(input_path)

    df = df.rename(columns={'jokeId': 'ID', 'jokeText': 'Body'})
    df['Title'] = pd.NA
    df['Category'] = pd.NA
    # df['Rating'] = pd.NA # Vine coloana de Rating din csv-ul de ratings.

    df_ratings = pd.read_csv(ratings_path)
    df_ratings = df_ratings.drop(columns=['userId'])
    df_ratings = df_ratings.rename(columns={'jokeId': 'ID', 'rating': 'Rating'})
    df_ratings = df_ratings.groupby('ID', as_index=False)['Rating'].mean()

    df = pd.merge(df, df_ratings, on='ID', how='left')

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_jester('Original-Datasets/Positive-Examples/jester/jester_items.csv', 'Original-Datasets/Positive-Examples/jester/jester_ratings.csv', 'Preprocessed-Datasets/Positive-Examples/jester/jester.csv')


def preprocess_dadjokes(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.rename(columns={'id': 'ID', 'title': 'Title', 'body': 'Body', 'category': 'Category', 'rating': 'Rating'})
    df['Body'] = df['question'] + '... ' + df['response']
    df = df.drop(columns=['question', 'response'])
    df['ID'] = pd.NA
    df['Title'] = pd.NA
    df['Category'] = pd.NA
    df['Rating'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_dadjokes('Original-Datasets/Positive-Examples/dadjokes/train.csv', 'Preprocessed-Datasets/Positive-Examples/dadjokes/train.csv')
# preprocess_dadjokes('Original-Datasets/Positive-Examples/dadjokes/test.csv', 'Preprocessed-Datasets/Positive-Examples/dadjokes/test.csv')


def break_big_jester_file(input_directory, file_name, output_directory, batch_size):
    df = pd.read_csv(input_directory + '/' + file_name + '.csv')

    batches = np.array_split(df, batch_size)

    for idx, batch in enumerate(batches):
        batch.to_csv(output_directory + '/' + file_name + '_' + str(idx) + '.csv', index=False)


# break_big_jester_file('Preprocessed-Datasets/Positive-Examples/jester', 'jester', 'Preprocessed-Datasets/Positive-Examples/jester', 18)


def preprocess_news_category_dataset(input_path, output_path):
    df = pd.read_json(input_path, lines=True)

    df = df.rename(columns={'headline': 'Title', 'category': 'Category', 'short_description': 'Body'})
    df = df.drop(columns=['link', 'authors', 'date'])
    df['ID'] = pd.NA
    df['Rating'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_news_category_dataset('Original-Datasets/Negative-Examples/News_Category_Dataset_v3/News_Category_Dataset_v3.json', 'Preprocessed-Datasets/Negative-Examples/News_Category_Dataset_v3/News_Category_Dataset_v3.csv')





