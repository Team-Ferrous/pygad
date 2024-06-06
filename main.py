# web service interfaces
import supabase       as S
import localStoragePy as localStorage

# tools
import os
import requests
import json
import glob
from tempfile import NamedTemporaryFile
from dotenv   import load_dotenv, dotenv_values 
import pathlib

# Artificial Machine Learning
import pygad
from merkly.mtree import MerkleTree
from typing import Callable

load_dotenv()

# accessing supabase
access_token       = ""  
url: str           = os.getenv("SUPABASE_URL")
key: str           = os.getenv("SUPABASE_KEY")
supabase: S.Client = S.create_client(url, key)

# Supabase Functions (Auth, Database, Datastore)
def query():
    data = supabase.auth.get_session()
    if data is not None:
        response = supabase.table('memories').select("associated_emotions").execute()
    return response

def insert(id:int, mem:[str]):
    d = supabase.auth.get_session()
    if d is not None:
        data, count = supabase.table('memories').insert({"id": id, "associated_memories": mem}).execute()

def update(id:int, mem:[str]):
    d = supabase.auth.get_session()
    if d is not None:
        data, count = supabase.table('memories').update({"associated_memories": mem}).eq('id', id).execute()

def register_user(email: str, password: str):
   supabase.auth.sign_up({
    "email":    email,
    "password": password
    })
   
def login_user(email, password):
    access_token = supabase.auth.sign_in_with_password({"email": email, "password": password})
    localStorage.setItem("access_token", supabase.auth.get_session())
    index = load_from_cloud_data_bucket('MatterBucket', f'{st.session_state.username.split("@")[0]}/conversation_master.pdf', st.session_state["model"])
    st.session_state.activate_chat = True

def logout_user():
    supabase.auth.sign_out()

def upload(bucket_name, destination, source):
    res  = supabase.storage.from_(bucket_name).update(destination, source)
    return res
    
def download(bucket_name, file_name, dl_path):
    with open(file_name, 'wb+') as f:
        res = supabase.storage.from_(bucket_name).download(dl_path)
        f.write(res)


'''
# X, Y, and Z location to set
default_cube.location = (0.0, 0.0, 0.0)
# Set the keyframe with that location, and which frame.
default_cube.keyframe_insert(data_path="location", frame=1)

# do it again!
default_cube.location = (3.0, 2.0, 1.0)
# setting it for frame 10
default_cube.keyframe_insert(data_path="location", frame=10)
'''

# choose any hash function that is of type (bytes, bytes) -> bytes
# my_hash_function: Callable[[bytes, bytes], bytes] = lambda x, y: x + y

# # create a Merkle Tree
# mtree = MerkleTree(['a', 'b', 'c', 'd'], my_hash_function)

# # show original input
# assert mtree.raw_leaves == ['a', 'b', 'c', 'd']

# # hashed leaves
# assert mtree.leaves == [b'a', b'b', b'c', b'd']

# # shorted hashed leaves
# assert mtree.short_leaves == [b'a', b'b', b'c', b'd']


y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44

my_hash_function: Callable[[bytes, bytes], bytes] = lambda x, y: x + y
function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output  = 44
mtree           = MerkleTree(function_inputs, my_hash_function)

#functions
def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

fitness_function = fitness_func
num_generations = 50
num_parents_mating = 4
sol_per_pop = 8
num_genes = len(function_inputs)
init_range_low = -2
init_range_high = 5
parent_selection_type = "sss"
keep_parents   = 1
crossover_type = "single_point"
mutation_type  = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = numpy.sum(numpy.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

with NamedTemporaryFile(dir='.', suffix='.txt') as f:
        f.write(prediction)