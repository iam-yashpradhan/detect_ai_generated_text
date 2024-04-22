from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np


pc = Pinecone(api_key="9f8be296-36ca-4a85-adf0-82939f72b2cf")
index = pc.Index('aidetection')
data = pd.read_csv('data.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

def encodeUpsert(data, start, iterations):
    chunk_size = 500
    start_idx = start
    num_iterations = iterations

    for iteration in range(num_iterations):
        end_idx = start_idx + chunk_size
        vectors_to_upsert = []

        for i in tqdm(range(start_idx, end_idx)):
            vectors_to_upsert.append({
                'id': str(data.loc[i, 'id']),
                'values': model.encode(data.loc[i, 'context']).tolist(),
                'metadata': {'text': data.loc[i, 'context']}
            })
        index.upsert(vectors=vectors_to_upsert)
        start_idx = end_idx

encodeUpsert(data, 240000, 120)

# Ends here

def querySimilarity(data):
    xq = model.encode(data).tolist()
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    return xc
    xc.dtype

query = 'The Pythagoras Theorem, also known as the Pythagorean Theorem, is a fundamental principle in geometry that describes the relationship between the sides of a right-angled triangle. It is named after the ancient Greek mathematician and philosopher, Pythagoras, who is credited with discovering this important theorem.'
querySimilarity(query)

xq = model.encode(query).tolist()
xc = index.query(vector=xq, top_k=5, include_metadata=True)
for i in xc.matches:
    print(i)

data = []
for response in xc.matches:
    data.append({'id': response['id'], 'score': response['score']})

data = pd.DataFrame(data)


aiData = pd.read_csv('/data/aiVector.csv')
humanData = pd.read_csv('/data/humanVector.csv')
collatedData= aiData.append(humanData)
def accMatrix(text):
    xq = model.encode(text).tolist()
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    simData = []
    for response in xc.matches:
        simData.append({'id': response['id'], 'score': response['score']})

    simData = pd.DataFrame(simData)

    simData['id'] = simData['id'].astype(int)
    count_ai = sum(value >= 150000 for value in simData['id'])
    count_human = sum(value < 150000 for value in simData['id'])
    if count_ai > count_human:
        collatedData['predicted'] = 'AI'
    else:
        collatedData['predicted'] = 'Human'

for i in collatedData['context']:
    accMatrix(i)

# # batch_size = 128
# # vector_limit = 100000
# # data['context'] = data['context'][:vector_limit]
#
# # i=2
# # for i in tqdm(range(0, len(trimD['context']))):
# #     # i_end = min(i + batch_size, len(data['context']))
# #     ids = [str(x) for x in range(i, len(trimD['context']))]
# #     metadatas = [{'text': text} for text in trimD['context']]
# #     xc = model.encode(trimD['context'])
# #     records = zip(ids, xc, metadatas)
# #     index.upsert(vectors=records)
# #
# #
# # print(index.describe_index_stats())
#
#
#
# # dataHuman = data[:2500]
# # dataAI = data[150000:152500]
# #
# # vectorHuman = createVectorDatabase(dataHuman)
# # vectorAI = createVectorDatabase(dataAI)
# #
# #
# # upsert_response = index.upsert(
# #     vectors=[
# #         {'id': str(vectorHuman['id']),
# #          'values': list(map(float, vectorHuman['vectors'])),
# #          'metadata': {'text': str(vectorHuman['context'])},
# #          }
# #
# #     ]
# # )
# #
# # index_info = pc.list_indexes()
# # dimension = index_info[0]['dimension']
# #
# #
# # index.upsert(ids = vectorHuman['id'].tolist(), vectors = vectorHuman['vectors'].tolist())
#
# # This Works: Really
# trimD = data[150000:151000]
# vectors_to_upsert = []
# for i in tqdm(range(len(trimD))):
#     vectors_to_upsert.append({
#         'id': str(trimD.loc[i, 'id']),
#         'values': model.encode(trimD.loc[i, 'context']).tolist(),
#         'metadata': {'text': trimD.loc[i, 'context']}
#     })
#
# index.upsert(vectors=vectors_to_upsert)
#
# # This works for all the PnCs
# # Create a function for this please
# chunk_size = 500
# start_idx = 151000
# end_idx = start_idx + chunk_size
# vectors_to_upsert = []
#
# for i in tqdm(range(start_idx, end_idx)):
#     vectors_to_upsert.append({
#             'id': str(data.loc[i, 'id']),
#             'values': model.encode(data.loc[i, 'context']).tolist(),
#             'metadata': {'text': data.loc[i, 'context']}
#         })
#
# index.upsert(vectors=vectors_to_upsert)

