from pymongo import MongoClient

import makeit.global_config as gc
client = MongoClient(gc.MONGO['path'], gc.MONGO[ 'id'], connect=gc.MONGO['connect'])
db = client['askcos_transforms']
SMARTS_DB = db['EFGs']
SMARTS_DB.remove({})

from EFGs import *

for i, fgroup in enumerate(fgroups):
	fgroups[i]['_id'] = i 

SMARTS_DB.insert(fgroups)