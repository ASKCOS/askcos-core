#!/usr/bin/env python3

import sys
import json
from . import reaction_context_predictor


predictor = reaction_context_predictor.ReactionContextRecommenderWLN()
#predictor = reaction_context_predictor.ReactionContextRecommenderFP()

fn = sys.argv[1]
fn_out = sys.argv[2]
f = open(fn, 'r')
f_out = open(fn_out, 'w')
print('input file: ', f.name)
print('output file: ', f_out.name)

for s in f:
    s = s.strip()
    #try:
    results = predictor.predict(smiles=s)
    #except Exception as e:
    #    print(e)
    #    results = None
    record = {'smiles':s, 'condition':results}
    f_out.write(json.dumps(record)+'\n')

f_out.close()
f.close()
