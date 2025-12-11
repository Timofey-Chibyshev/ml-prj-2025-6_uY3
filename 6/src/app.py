import sys
import os
from preprocess import generate_model_input
from postprocess import postprocess_model_output
from iam_sentences_crnn.data_loader import loader

import Levenshtein as lev

calc_example_cer = False
example = "A MOVE to stop Mr. Gaitskell from\nnominating any more Labour life Peers\nis to be made at a meeting at Labour\nMPs tomorrow. Mr. Michael Foot has put down a resolution on the subject\nand he is to be backed by Mr. Will\nGriffiths, MP for Manchester Exchange."

if len(sys.argv) <= 1:
    print("Usage: python app.py my-image.png")

path = sys.argv[1]

if len(sys.argv) >= 2 and sys.argv[2] == "--example":
    calc_example_cer = True;

try:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file exists: {path}")

    input = generate_model_input(path)
    model = loader.load_model()

    output = model(input)
    output = postprocess_model_output(output)
    output = '\n'.join(output)

    print(f"Recognition result:\n{output}")

    if calc_example_cer:
        cer = lev.distance(output, example) / len(example) * 100.0
        print(f"\nCER = {cer:.2f}%, Accuracy = {100.0 - cer:.2f}%")

except Exception as e:
    print(f"Error! {e}")
