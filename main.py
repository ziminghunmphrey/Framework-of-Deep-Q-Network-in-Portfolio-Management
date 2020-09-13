import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
from coordinator import Coordinator
from config import tuned_config


model = Coordinator(tuned_config, 'name')
model.train()
model.restore('name-step')
