import io
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd
import argparse

def get_text(image_file, symbol_search=None):
    """Get text from images"""
    client = vision.ImageAnnotatorClient()

    bounds = []

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    df = pd.DataFrame()

    # Collect specified feature bounds by enumerating all document features
    for n_page, page in enumerate(document.pages):
        for n_block, block in enumerate(page.blocks):
            for n_paragraph, paragraph in enumerate(block.paragraphs):
                for n_word, word in enumerate(paragraph.words):

                    symbol_search_find = False
                    n_symbol_search_find = None

                    for n_symbol, symbol in enumerate(word.symbols):
                        if symbol.text == symbol_search:
                            
                            print('find simbol')
                            symbol_search_find = True
                            n_symbol_search_find = n_symbol

                    df_temp = pd.DataFrame({
                                            'n_page': n_page,
                                            'page_height': [page.height],
                                            'page_width': [page.width],
                                            'n_block': [n_block],
                                            'block_vert': [[(vertex.x, vertex.y) for vertex in block.bounding_box.vertices]],
                                            'block_confidence': block.confidence,
                                            'n_paragraph': n_paragraph,
                                            'paragraph_vert': [[(vertex.x, vertex.y) for vertex in paragraph.bounding_box.vertices]],
                                            'paragraph_confidence': paragraph.confidence,
                                            'n_word': n_word,
                                            'word_vert': [[(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]],
                                            "word_confidence": word.confidence,
                                            "text": [[symbol.text for symbol in word.symbols]],
                                            "symbol_search_pos": symbol_search_find,
                                            "n_symbol_search_find": n_symbol_search_find,
                                            })

                    df = df.append(df_temp, ignore_index=True)


     
    return df


if __name__ == '__main__':

    # delete
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/scors/Credentials/Raxar-12e4ceccdc19.json"
    # -----

    parser = argparse.ArgumentParser()
    parser.add_argument('detect_file', help='The image for text detection.')
    parser.add_argument('output_file_name', help='Name of the output file to save')
    parser.add_argument('-detect_symbol', help='Optional detect symbol', default=None)
    args = parser.parse_args()

    df = get_text(args.detect_file, args.detect_symbol)
    df.to_csv(args.output_file_name + '.csv')