from time import time
import math
from maxidom_hack.utils.clients import EmbeddingClient
from maxidom_hack.db.loader import DBLoader

init_t = time()
embed_client = EmbeddingClient()
print(f'LLMClient initialized in {time() - init_t:.2f} seconds')

db_t = time()
db_loader = DBLoader(embed_client=embed_client)
print(f'DBLoader initialized in {time() - init_t:.2f} seconds')
db_loader.delete_and_load('/Users/dolor.src/Downloads/rel_data_annotated.json')