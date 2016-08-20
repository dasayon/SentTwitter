import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union

train_text=state_union.raw("2005-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")

#initialise custom tokenizer to be trained on sample sample_text
custom_tokenizer=PunktSentenceTokenizer(train_text);

#tokenize into sentences
tokenized=custom_tokenizer.tokenize((sample_text))

def process_content():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i);
            tagged=nltk.pos_tag(words)
            namedEnt=nltk.ne_chunk(tagged,binary=True)
            namedEnt.draw()
    except Exception as e:
        print(str(e))


process_content()

'''
      chunkgrammar=r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            chunkParser=nltk.RegexpParser(chunkgrammar)

            chunked=chunkParser.parse(tagged)
            ##print(chunked)
            chunked.draw()
'''




