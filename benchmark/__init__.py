import os
import sys
import torch
# from .utils import get_prompt_from_filename, init_submodules, save_json, load_json
# from .distributed import get_rank, print0

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
import benchmark.utils


class benchmark:
    def __init__():
        pass