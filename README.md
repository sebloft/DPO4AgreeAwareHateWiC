# DPO4AgreeAwareHateWiC
Code for 'Using LLMs and Preference Optimization for Agreement-Aware HateWiC Classification'


## Running the Code

### Training

To train a model, e.g. the base model (OpenRLHF/Llama-3-8b-sft-mixture) on task (variation) 1 and the first fold, run:

```bash
bash train.sh --model base --variation 1 --fold 1
```

*There are also the 'instruct' model (OpenRLHF/Llama-3-8b-rlhf-100k), a task '2' and 9 more folds.*

### Testing
After training, testing the same model can be done with the following command:

```bash
python test_model.py --variation 1 --fold 1
```

The average over all folds can then be computed in the *average.ipynb* notebook.
