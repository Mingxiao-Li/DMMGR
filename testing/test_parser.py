import spacy
from SceneGraphParser import sng_parser
from pprint import pprint

if __name__ == "__main__":
    nlp = spacy.load("en")
    sentence = "the relation between man and the object that is related to road in the image"
    doc = nlp(sentence)
   # for chunk in doc.noun_chunks:
   #     print(chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text)
   #     print("-"*40)
    graph = sng_parser.parse(sentence)
    pprint(graph)
    #for token in doc:
    #    if token.pos_ in ["NOUN", "NN"]:
    #        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #        token.shape_, token.is_alpha, token.is_stop)

    #for token in doc:
    #    print(token.text, token.dep_, token.head.text, token.head.pos_,
    #          [child for child in token.children])