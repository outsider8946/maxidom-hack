from time import time
import math
from maxidom_hack.llm_client import LLMClient
from maxidom_hack.db.loader import DBLoader

init_t = time()
llm_client = LLMClient()
print(f'LLMClient initialized in {time() - init_t:.2f} seconds')

db_t = time()
db_loader = DBLoader(llm_client=llm_client)
print(f'DBLoader initialized in {time() - init_t:.2f} seconds')
db_loader.delete_and_load('/Users/dolor.src/Downloads/rel_data_annotated.json')