import pandas as pd
import pickle
import argparse
import get_features, get_text


def predict_price(image_file, model, symbol_search='$'):

    df = get_text.get_text(image_file, symbol_search)
    df = get_features.get_features(df)

    features = [
                'block_confidence',
                'paragraph_confidence',
                'word_confidence',
                'block_weigh',
                'paragraph_weigh',
                'word_weigh',
                'rel_word_block_area',
                'rel_word_parag_area',
                'rel_parag_block_area',
                'prev_symbol',
                'next_symbol',
                'text_type',
                'prev_text_type',
                'next_text_type',
                'text_len',
                'is_a_symbol',
                'number_type',
                ]
    X = df[features].copy()
    
    y_pred = model.predict(X.values)
    df['predict'] = y_pred

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
    args = parser.parse_args()

    # predict price
    df = predict_price(args.detect_file, model)

    # print results
    print('Price:', df[df['predict']==1]['number'].values)
    print()
    print('Text:')
    print( '_'.join(df['text_join'].values))

    df.to_csv(args.output_file_name + '.csv')