import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import re

def area(x):
    
    return Polygon(x).area


def join_list(x):
    
    return ''.join(x)


def text_type(x):
    
    try:
        float(x)
        return 0

    except ValueError:

        return 1


def extrac_numbers(x):
    
    return re.sub('[^0-9,.]', "", x)


def number_type(x):
    
    try:
        float(x)
        return 1

    except ValueError:
        try:
            float(x.replace(',', '.'))

            return 1
    
        except ValueError:
            return 0
        
        
def number(x):
    
    try:
        return float(x)

    except ValueError:
        try:
            return float(x.replace(',', '.'))
    
        except ValueError:
            return 0


def get_features(df):
    
    # area
    columns = ['block_vert',
               'paragraph_vert',
               'word_vert',
              ]
    for col in columns:
        df[col.replace('vert', 'area')] = df[col].apply(area)
        
    # weighted area
    columns = ['block_area',
               'paragraph_area',
               'word_area',
              ]
    for col in columns:
        df[col.replace('area', 'weigh')] = df[col].divide(df[col].max())
        
    # relative area
    df.loc[:, 'rel_word_block_area'] = df['word_area'].divide(df['block_area'])
    df.loc[:, 'rel_word_parag_area'] = df['word_area'].divide(df['paragraph_area'])
    df.loc[:, 'rel_parag_block_area'] = df['paragraph_area'].divide(df['block_area'])
    
    # join text
    df.loc[:, 'text_join'] = df['text'].apply(join_list)
    
    # prev symbol
    df.loc[:, 'prev_symbol'] = np.where(df['symbol_search_pos'].shift(-1), 1, 0)
    df.loc[:, 'prev_symbol'] = df.loc[:, 'prev_symbol'].fillna(0)
    
    # next symbol
    df.loc[:, 'next_symbol'] = np.where(df['symbol_search_pos'].shift(1), 1, 0)
    df.loc[:, 'next_symbol'] = df.loc[:, 'next_symbol'].fillna(0)
    
    # text type
    df.loc[:, 'text_type'] = df['text_join'].apply(text_type)
    
    # prev text type
    df.loc[:, 'prev_text_type'] = df['text_type'].shift(1)
    df.loc[:, 'prev_text_type'] = df.loc[:, 'prev_text_type'].fillna(0)
    
    # next text tpye
    df.loc[:, 'next_text_type'] = df['text_type'].shift(-1)
    df.loc[:, 'next_text_type'] = df.loc[:, 'next_text_type'].fillna(0)
    
    # text len
    df.loc[:, 'text_len'] = df['text_join'].str.len()
    
    # is symbol
    df.loc[:, 'is_a_symbol'] = np.where(df['symbol_search_pos'], 1, 0)
    
    # extract numbers
    df.loc[:, 'text_numbers'] = df['text_join'].apply(extrac_numbers)
    
    # number type
    df.loc[:, 'number_type'] = df['text_numbers'].apply(number_type)
    
    # number
    df.loc[:, 'number'] = df['text_join'].apply(number)
    
    return df