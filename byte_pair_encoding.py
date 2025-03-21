from collections import defaultdict
import re

corpus = "Hello world!"
corpus = re.findall("\w+", corpus.lower())

vocabulary = defaultdict(int)
charset = set()

word = corpus[0]
word_with_marker = "_" + word
characters = list(word_with_marker)
charset.update(characters)
tokenized_word = " ".join(characters)
vocabulary[tokenized_word] += 1
