
import os
import sys

import pandas as pd

from gutenberg.src import cleanup

# python3 generate_corpus.py authors.txt gutenberg/metadata/metadata.csv gutenberg/data/raw/ ./saved_texts ./to_test


def main():

    authors_path = sys.argv[1]
    metadata_path = sys.argv[2]
    raw_path = sys.argv[3]
    save_path = sys.argv[4]
    test_path = sys.argv[5]

    with open(authors_path, 'r') as authors_file:
        authors = [x.strip() for x in authors_file.readlines()]

    df = pd.read_csv(metadata_path)
    df = df.sort_values('id') # multiplos arquivos por

    for author in authors:
        subset = df[(df['author'] == author)
                    & df['language'].apply(lambda x: ('en' in x) if not pd.isnull(x) else False)]
        # remove duplicated files and missing files
        subset = subset[subset.apply(lambda row: os.path.exists(os.path.join(raw_path, row['id'] + '_raw.txt')), axis=1)]

        # subset for testing
        sample_to_save = subset.sample(n=30)
        subset = subset.drop(sample_to_save.index, axis=0)

        with open(os.path.join(save_path, '{}_all.txt'.format(author.replace(' ', '_'))), 'w') as text_file:
            for ind, row in subset.iterrows():
                #if os.path.exists(os.path.join(raw_path, row['id'] + '_raw.txt')):
                with open(os.path.join(raw_path, row['id'] + '_raw.txt'), 'r', encoding='utf-8', errors='ignore') \
                        as book_file:
                    book = book_file.read()
                    book = cleanup.strip_headers(book)
                    text_file.write(book)

        for (ind, row), num in zip(sample_to_save.iterrows(), range(30)):
            with open(os.path.join(test_path, '{}_{}.txt'.format(author.replace(' ', '_'), num)), 'w') as text_file:
                #if os.path.exists(os.path.join(raw_path, row['id'] + '_raw.txt')):
                with open(os.path.join(raw_path, row['id'] + '_raw.txt'), 'r', encoding='utf-8', errors='ignore') \
                        as book_file:
                    book = book_file.read()
                    book = cleanup.strip_headers(book)
                    text_file.write(book)


if __name__ == '__main__':
    main()
