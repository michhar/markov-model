from text import TextMarkovModel
import pickle
import os

with open('data/text_shorter.txt', 'r') as file:
    text = file.read()

modelfile = 'model.pkl'
if os.path.exists(modelfile):
    with open(modelfile, 'rb') as f:
        mm = pickle.load(f)
else:
    mm = TextMarkovModel(text)
    # This may take some time so we save as pickle file
    mm.train()
    with open('model.pkl', 'wb') as f:
        pickle.dump(mm, f)

print(' '.join(mm.generate_chain("understand", 25)))
