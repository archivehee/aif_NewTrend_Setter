import os
import world
import dataloader
import model
import utils
from pprint import pprint

# resolve InitGrounder project root to locate inter_cdr data
_code_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_code_dir, "..", "..", ".."))
_inter_cdr_root = os.path.join(_project_root, "data")

if world.dataset == 'Home_and_Kitchen':
    dataset_path = os.path.join(_inter_cdr_root, "Home_and_Kitchen")
    dataset = dataloader.Home_Kit(config=world.config, path=dataset_path)
else:
    raise ValueError(f"Unsupported dataset: {world.dataset}")

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}
