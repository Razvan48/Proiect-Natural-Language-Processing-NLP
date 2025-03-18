import pandas as pd



def preprocess_shortjokes(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.rename(columns={'ID': 'ID', 'Joke': 'Body'})
    df['Title'] = pd.NA
    df['Category'] = pd.NA
    df['Rating'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False) # index=False nu adauga index la randuri


# preprocess_shortjokes('Original-Datasets/shortjokes/shortjokes.csv', 'Preprocessed-Datasets/shortjokes/shortjokes.csv')


def preprocess_reddit_jokes(input_path, output_path):
    df = pd.read_json(input_path)

    df = df.rename(columns={'id': 'ID', 'title': 'Title', 'body': 'Body', 'score': 'Rating'})
    df['Category'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_reddit_jokes('Original-Datasets/joke-dataset/reddit_jokes.json', 'Preprocessed-Datasets/joke-dataset/reddit_jokes.csv')


def preprocess_stupidstuff(input_path, output_path):
    df = pd.read_json(input_path)

    df = df.rename(columns={'id': 'ID', 'category': 'Category', 'body': 'Body', 'rating': 'Rating'})
    df['Title'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_stupidstuff('Original-Datasets/joke-dataset/stupidstuff.json', 'Preprocessed-Datasets/joke-dataset/stupidstuff.csv')


def preprocess_wocka(input_path, output_path):
    df = pd.read_json(input_path)

    df = df.rename(columns={'id': 'ID', 'title': 'Title', 'category': 'Category', 'body': 'Body'})
    df['Rating'] = pd.NA

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_wocka('Original-Datasets/joke-dataset/wocka.json', 'Preprocessed-Datasets/joke-dataset/wocka.csv')


def preprocess_jester(input_path, ratings_path, output_path):
    df = pd.read_csv(input_path)

    df = df.rename(columns={'jokeId': 'ID', 'jokeText': 'Body'})
    df['Title'] = pd.NA
    df['Category'] = pd.NA
    # df['Rating'] = pd.NA # Vine coloana de Rating din csv-ul de ratings.

    df_ratings = pd.read_csv(ratings_path)
    df_ratings = df_ratings.drop(columns=['userId'])
    df_ratings = df_ratings.rename(columns={'jokeId': 'ID', 'rating': 'Rating'})

    df = pd.merge(df, df_ratings, on='ID', how='left')

    df = df[['ID', 'Title', 'Category', 'Body', 'Rating']]

    df.to_csv(output_path, index=False)


# preprocess_jester('Original-Datasets/jester/jester_items.csv', 'Original-Datasets/jester/jester_ratings.csv', 'Preprocessed-Datasets/jester/jester.csv')


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


# preprocess_dadjokes('Original-Datasets/dadjokes/train.csv', 'Preprocessed-Datasets/dadjokes/train.csv')
# preprocess_dadjokes('Original-Datasets/dadjokes/test.csv', 'Preprocessed-Datasets/dadjokes/test.csv')


