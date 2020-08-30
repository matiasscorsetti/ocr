import pandas as pd
import pickle
import argparse
import get_features, get_text


def predict_price(image_file, model, symbol_search=None):

    df = get_text.get_text(image_file, symbol_search)
    df = get_features.get_features(df)

    features = ['page_height',
                'page_width',
                'block_confidence',
                'paragraph_confidence',
                'word_confidence',
                'block_area',
                'paragraph_area',
                'word_area',
                'prev_symbol',
                'text_type',
                'prev_text_type',
                'next_text_type',
                'rel_word_block_area',
                'rel_word_parag_area',
                'rel_parag_block_area',
                'text_len',
                'number_type',
                ]
    X = df[features].copy()
    
    y = model.predict(X.values)
    
    df['predict'] = y

    return df




if __name__ == '__main__':

    # delete
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/scors/Credentials/Raxar-12e4ceccdc19.json"
    # -----

    with open('../models/classification_model.pkl', 'rb') as f:
        classification_model = pickle.load(f)
    model = classification_model['model']

    parser = argparse.ArgumentParser()
    parser.add_argument('detect_file', help='The image for text detection.')
    parser.add_argument('output_file_name', help='Name of the output file to save')
    parser.add_argument('-detect_symbol', help='Optional detect symbol', default=None)
    args = parser.parse_args()

    # predict price
    df = predict_price(args.detect_file, model, args.detect_symbol)

    # print price
    print('Price:', df[df['predict']==1]['text_join'].values)

    df.to_csv(args.output_file_name + '.csv')