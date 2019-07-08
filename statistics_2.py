
import os
import sys

import pandas as pd

from gutenberg.src import cleanup

#python3 statistics_2.py authors.txt gutenberg/metadata/metadata.csv gutenberg/data/raw/


def main():

    authors_path = sys.argv[1]
    metadata_path = sys.argv[2]
    raw_path = sys.argv[3]

    with open(authors_path, 'r') as authors_file:
        authors = [x.strip() for x in authors_file.readlines()]

    df = pd.read_csv(metadata_path)
    df = df.sort_values('id')

    selected_authors = {
        'author': [],
        'title': [],
        'id': [],
        'tokens': [],
        'chars': []
    }

    for author in authors:
        subset = df[df['author'] == author]
        for ind, row in subset.iterrows():
            if os.path.exists(os.path.join(raw_path, row['id'] + '_raw.txt')):
                with open(os.path.join(raw_path, row['id'] + '_raw.txt'), 'r', encoding='utf-8', errors='ignore') as book_file:
                    book = book_file.read()

                    book = cleanup.strip_headers(book)

                    book_chars = len(book)
                    book = book.split()
                    book_size = len(book)

                    selected_authors['author'].append(row['author'])
                    selected_authors['title'].append(row['title'])
                    selected_authors['id'].append(row['id'])
                    selected_authors['tokens'].append(book_size)
                    selected_authors['chars'].append(book_chars)

    df_select = pd.DataFrame(selected_authors)

    df_select.to_csv('authors_tokens.csv', index=False)


if __name__ == '__main__':
    main()
