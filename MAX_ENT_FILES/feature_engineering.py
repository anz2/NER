import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


def read_text(path):
    with open(path, 'r') as file:
        return file.read()


def write_text(path, text):
    with open(path, 'w') as file:
        file.write(text)


def get_dictionary_from_text(text, test=False):
    if not test:
        data = {'token': [], 'pos': [], 'BIO': []}
        for line in text.split('\n'):
            if line == '':
                data['token'].append('')
                data['pos'].append('')
                data['BIO'].append('')
            else:
                tok, pos, chk = line.split('\t')
                data['token'].append(tok)
                data['pos'].append(pos)
                data['BIO'].append(chk)
    else:
        data = {'token': [], 'pos': []}
        for line in text.split('\n'):
            if line == '':
                data['token'].append('')
                data['pos'].append('')
            else:
                tok, pos = line.split('\t')
                data['token'].append(tok)
                data['pos'].append(pos)

    return data


def shift_and_pad(col, pad=0):
    if pad > 0:
        col = np.concatenate([[''] * pad, col[:-pad]])
    elif pad == 0:
        pass
    elif pad < 0:
        col = np.concatenate([col[-pad:], [''] * (-pad)])

    return col


def save_df_as_text(df, path):
    df.to_csv(path, sep='\t', header=False, index=False)


def add_col_names_in_cols(df):
    cols = df.columns
    df.loc[df.token == '', :] = '<EMPTY_LINE>'
    for col in cols:
        df[col] = df[col].apply(lambda x: col + '=' + x if x != '<EMPTY_LINE>' else '')

    return df


if __name__ == '__main__':
    generate_train_features, generate_test_features = True, True

    if generate_train_features:
        train_text = read_text('../WSJ_CHUNK_FILES/WSJ_02-21.pos-chunk')
        train_dict = get_dictionary_from_text(train_text)
        train_df = pd.DataFrame.from_dict(train_dict)

        porter_stemmer = PorterStemmer()
        wordnet_lemmatizer = WordNetLemmatizer()

        features_df = train_df.iloc[:, :2]
        features_df['lemma'] = features_df.token.apply(wordnet_lemmatizer.lemmatize)
        features_df['stem'] = features_df.token.apply(porter_stemmer.stem)

        features_df['prev_token'] = shift_and_pad(features_df.token.values, +1)
        features_df['prev_pos'] = shift_and_pad(features_df.pos.values, +1)
        features_df['prev_lemma'] = shift_and_pad(features_df.lemma.values, +1)
        features_df['prev_stem'] = shift_and_pad(features_df.stem.values, +1)
        features_df.loc[features_df.token == '', ['prev_token', 'prev_pos', 'prev_lemma', 'prev_stem']] = ''

        features_df['prev_prev_token'] = shift_and_pad(features_df.token.values, +2)
        features_df['prev_prev_pos'] = shift_and_pad(features_df.pos.values, +2)
        features_df['prev_prev_lemma'] = shift_and_pad(features_df.lemma.values, +2)
        features_df['prev_prev_stem'] = shift_and_pad(features_df.stem.values, +2)
        features_df.loc[
            features_df.prev_token == '', ['prev_prev_token', 'prev_prev_pos', 'prev_prev_lemma',
                                           'prev_prev_stem']] = ''
        features_df.loc[
            features_df.token == '', ['prev_prev_token', 'prev_prev_pos', 'prev_prev_lemma', 'prev_prev_stem']] = ''

        features_df['next_token'] = shift_and_pad(features_df.token.values, -1)
        features_df['next_pos'] = shift_and_pad(features_df.pos.values, -1)
        features_df['next_lemma'] = shift_and_pad(features_df.lemma.values, -1)
        features_df['next_stem'] = shift_and_pad(features_df.stem.values, -1)
        features_df.loc[features_df.token == '', ['next_token', 'next_pos', 'next_lemma', 'next_stem']] = ''

        features_df['next_next_token'] = shift_and_pad(features_df.token.values, -2)
        features_df['next_next_pos'] = shift_and_pad(features_df.pos.values, -2)
        features_df['next_next_lemma'] = shift_and_pad(features_df.lemma.values, -2)
        features_df['next_next_stem'] = shift_and_pad(features_df.stem.values, -2)
        features_df.loc[
            features_df.next_token == '', ['next_next_token', 'next_next_pos', 'next_next_lemma',
                                           'next_next_stem']] = ''
        features_df.loc[
            features_df.token == '', ['next_next_token', 'next_next_pos', 'next_next_lemma', 'next_next_stem']] = ''

        features_df['BIO'] = train_df.BIO.values

        features_df = add_col_names_in_cols(features_df)
        save_df_as_text(features_df, '../WSJ_CHUNK_FILES/training.feature')

    if generate_test_features:
        test_text = read_text('../WSJ_CHUNK_FILES/WSJ_24.pos')
        test_dict = get_dictionary_from_text(test_text, True)
        test_df = pd.DataFrame.from_dict(test_dict)

        porter_stemmer = PorterStemmer()
        wordnet_lemmatizer = WordNetLemmatizer()

        features_df = test_df.iloc[:, :2]
        features_df['lemma'] = features_df.token.apply(wordnet_lemmatizer.lemmatize)
        features_df['stem'] = features_df.token.apply(porter_stemmer.stem)

        features_df['prev_token'] = shift_and_pad(features_df.token.values, +1)
        features_df['prev_pos'] = shift_and_pad(features_df.pos.values, +1)
        features_df['prev_lemma'] = shift_and_pad(features_df.lemma.values, +1)
        features_df['prev_stem'] = shift_and_pad(features_df.stem.values, +1)
        features_df.loc[features_df.token == '', ['prev_token', 'prev_pos', 'prev_lemma', 'prev_stem']] = ''

        features_df['prev_prev_token'] = shift_and_pad(features_df.token.values, +2)
        features_df['prev_prev_pos'] = shift_and_pad(features_df.pos.values, +2)
        features_df['prev_prev_lemma'] = shift_and_pad(features_df.lemma.values, +2)
        features_df['prev_prev_stem'] = shift_and_pad(features_df.stem.values, +2)
        features_df.loc[
            features_df.prev_token == '', ['prev_prev_token', 'prev_prev_pos', 'prev_prev_lemma',
                                           'prev_prev_stem']] = ''
        features_df.loc[
            features_df.token == '', ['prev_prev_token', 'prev_prev_pos', 'prev_prev_lemma', 'prev_prev_stem']] = ''

        features_df['next_token'] = shift_and_pad(features_df.token.values, -1)
        features_df['next_pos'] = shift_and_pad(features_df.pos.values, -1)
        features_df['next_lemma'] = shift_and_pad(features_df.lemma.values, -1)
        features_df['next_stem'] = shift_and_pad(features_df.stem.values, -1)
        features_df.loc[features_df.token == '', ['next_token', 'next_pos', 'next_lemma', 'next_stem']] = ''

        features_df['next_next_token'] = shift_and_pad(features_df.token.values, -2)
        features_df['next_next_pos'] = shift_and_pad(features_df.pos.values, -2)
        features_df['next_next_lemma'] = shift_and_pad(features_df.lemma.values, -2)
        features_df['next_next_stem'] = shift_and_pad(features_df.stem.values, -2)
        features_df.loc[
            features_df.next_token == '', ['next_next_token', 'next_next_pos', 'next_next_lemma',
                                           'next_next_stem']] = ''
        features_df.loc[
            features_df.token == '', ['next_next_token', 'next_next_pos', 'next_next_lemma', 'next_next_stem']] = ''

        features_df = add_col_names_in_cols(features_df)
        save_df_as_text(features_df, '../WSJ_CHUNK_FILES/test.feature')
