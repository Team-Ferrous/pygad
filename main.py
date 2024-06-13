# web service interfaces
import supabase       as S

# tools
import os
import yaml
from tempfile import NamedTemporaryFile
from dotenv   import load_dotenv, dotenv_values 

# machine learning components
import pygad # Artificial Machine Learning
from merkly.mtree import MerkleTree   
from typing import Callable
import numpy
from PIL import Image
import argparse

load_dotenv()
with open('GA_Options.yaml', 'r') as f:
    GAData = yaml.load(f, Loader=yaml.SafeLoader)

# accessing supabase
access_token       = ""  
url: str           = os.getenv("SUPABASE_URL")
key: str           = os.getenv("SUPABASE_KEY")
supabase: S.Client = S.create_client(url, key)

parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number", type=int)
args = parser.parse_args()
print(args.square**2)

# Supabase Functions (Auth, Database, Datastore)
def query():
    data = supabase.auth.get_session()
    if data is not None:
        response = supabase.table('metadata').select("associated_emotions").execute()
    return response

def insert(id:int, mem:[str]):
    d = supabase.auth.get_session()
    if d is not None:
        data, count = supabase.table('metadata').insert({"id": id, "associated_memories": mem}).execute()

def update(id:int, mem:[str]):
    d = supabase.auth.get_session()
    if d is not None:
        data, count = supabase.table('metadata').update({"associated_memories": mem}).eq('id', id).execute()

def register_user(email: str, password: str):
   supabase.auth.sign_up({
    "email":    email,
    "password": password
    })
   
def login_user(email, password):
    access_token = supabase.auth.sign_in_with_password({"email": email, "password": password})
    localStorage.setItem("access_token", supabase.auth.get_session())
    index = load_from_cloud_data_bucket('MatterBucket', f'{st.session_state.username.split("@")[0]}/conversation_master.pdf', st.session_state["model"])
  
def logout_user():
    supabase.auth.sign_out()

def upload(bucket_name, destination, source):
    res  = supabase.storage.from_(bucket_name).update(destination, source)
    return res
    
def download(bucket_name, file_name, dl_path):
    with open(file_name, 'wb+') as f:
        res = supabase.storage.from_(bucket_name).download(dl_path)
        f.write(res)

#Example Case:
#y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
#where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44

mhash_function: Callable[[bytes, bytes], bytes] = lambda x, y: x + y
function_inputs = [4, -2, 3.5, 5, -11,-4.7]
desired_output  = 44
bytes_inputs    = ['a', 'b', 'c', 'd']
mtree           = MerkleTree(bytes_inputs, mhash_function)

#functions
def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

input_image  = []
target_image = Image.open("target.jpg")  # numpy array of an image 
fitness_function   = fitness_func
num_genes = len(function_inputs)

#use DWT? or DCT?
for x in GAData['NUM_ITERATIONS']:
    ga_instance = pygad.GA(num_generations=GAData['NUM_GENERATIONS'],
                        num_parents_mating=GAData['NUM_PARENTS_MATING'],
                        fitness_func=fitness_function,
                        sol_per_pop=GAData['SOL_PER_POP'],
                        num_genes=num_genes,
                        init_range_low=GAData['INIT_RANGE_LOW'],
                        init_range_high=GAData['INIT_RANGE_HIGH'],
                        parent_selection_type=GAData['PARENT_SELECTION_TYPE'],
                        keep_parents=GAData['KEEP_PARENTS'],
                        crossover_type=GAData['CROSSOVER_TYPE'],
                        mutation_type=GAData['MUTATION_TYPE'],
                        mutation_percent_genes=GAData['MUTATION_PERCENT_GENES'])
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = numpy.sum(numpy.array(function_inputs)*solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    img = Image.open(sys.argv[1]).convert('L')

    im = numpy.array(img)
    fft_mag = numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(im)))

    visual = numpy.log(fft_mag)
    visual = (visual - visual.min()) / (visual.max() - visual.min())

    input_image = Image.fromarray((prediction * 255).astype(numpy.uint8))
    input_image.save('out.bmp')

    with open("predictions.txt", "w") as file1:
            file1.write(f"Solution Index:        {solution_idx}\n")
            file1.write(f"Best Soln. Parameters: {solution}\n")
            file1.write(f"Best Prediction:       {prediction}")

    with open("predictions_mtree.txt", "w") as file1:
            file1.write(f"{mtree.leaves}")

        
        
    #save results to .txt file(s)
    #leaves = mtree.leaves
    #with NamedTemporaryFile(dir='.', suffix='.txt') as f:
    #with NamedTemporaryFile(dir='.', suffix='.txt') as r:

                #r.write(leaves)
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
