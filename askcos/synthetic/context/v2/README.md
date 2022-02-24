# Prerequisite
- Tensorflow 2.x
- numpy
- rdkit >= 2019
- python >= 3.6

# Evaluate
```
# examples
python -m askcos.synthetic.context.v2.reaction_context_predictor.reaction_context_predictor

# one reaction SMILES per line in input.txt
python -m askcos.synthetic.context.v2.reaction_context_predictor.evaluate input.txt output.txt
```
