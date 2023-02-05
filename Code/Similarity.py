#Write some lines to encode (sentences 0 and 2 are both ideltical):
sen = [
"Text to measure similarity with",
"Text for which you want to measure similarity"

]
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('SPECTER ')
#Encoding:
sen_embeddings = model.encode(sen)
sen_embeddings.shape

from sklearn.metrics.pairwise import cosine_similarity
#let's calculate cosine similarity for sentence 0:
print(cosine_similarity(
    [sen_embeddings[0]],
    sen_embeddings[1:]
))
