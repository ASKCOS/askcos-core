# Make-It (makeit)
### Description
Prediction of chemical reactivity

### Requirements
The following is a list of non-default python packages known to be required by makeit:
- numpy
- rdkit
- keras
- cpickle
- pymongo
- cirpy (currently unused)

These can each be installed through
<code>pip install [package name]</code>. In addition, OPSIN look-up requires java to run.

### Workflow
#### 1. Prepare data sets
To avoid interacting with the database directly, any information relevant for a certain task should be pre-parsed and resaved as a .json dump. This is done by running <code>python makeit/utils/generate_data_subsets.py "data type" [max # records]</code> with the appropriate arguments. Currently, there are three available types of data:

- <code>"chemical_names_with_mws"</code> : each element in the dumped list consists of <code>[str(name), float(mol_weight)]</code>
- <code>"reactions_2reac_1prod"</code> : each element in the dumped list consists of <code>[str(A_name), str(B_name), str(C_name), float(yield)]</code>
- <code>"chemical_names"</code> : each element in the dumped list consists of <code>str(name)</code>

#### 2. Build tokenizer
The tokenizer must be fit before it can be used, so that words can be scored and ranked by the frequency of their occurrence in the corpus. For speed, building the tokenizer requires a json file which consists of a list of chemical names. This can be generated using the <code>"chemical_names"</code> option in <code>generate_data_subsets.py</code>. 

Tokenizers can be generated by running <code>python makeit/main/tokenize_chemicals.py "data_file.json" [max # vocab]</code>. This script will save the tokenizer to the same location every time, so the previous tokenizer will be over-written. It should not be necessary to run this script very often, however. The fitted tokenizer is saved using <code>cPickle</code>.

#### 3. Build model (ongoing)
For each different model (e.g., different model structure, sets of inputs, sets of outputs), there should be a different python file. Models are built and trained within the same script, because model construction takes such a short time compared to training. One such file can be run with <code>python makeit/main/neural_name_to_mw.py "tokenizer.cpickle" "data.json"</code>. The path to the tokenizer's cPickle back-up as well as the path to the reformatted database data are required.

#### 4. Visualize performance (ongoing)
