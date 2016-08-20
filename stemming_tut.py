from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
ps=PorterStemmer()

example_words=["pythoning","pythonly","pythoned","pythonising","pythoner"]

#for w in example_words:
#   print(ps.stem(w))

new_text="So I was running down the road trying to loosen my load I've got seven women on my mind. Four that wanna own me. Two that wanna stone me.One says she's a friend of mine"

tokenised=word_tokenize(new_text)

for w in tokenised:
    print(ps.stem(w))



