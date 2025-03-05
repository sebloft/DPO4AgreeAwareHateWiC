# DPO4AgreeAwareHateWiC
Code for 'Using LLMs and Preference Optimization for Agreement-Aware HateWiC Classification'


## Running the Code


### Training everything
To train everything the script *train_all.sh* should be used:

```bash
bash train_all.sh
```
There is also the possibility to train only on a subset of the folds e.g. only on fold one, three and five:
```bash
bash train_all.sh --folds 1 3 5
```

### Testing everything
To test all models base and fine-tuned models run:
```bash
bash test_all.sh
```
Again with the possibility to run only on a subset of the folds:
```bash
bash test_all.sh --folds 1 3 5
```

The average over all folds can then be computed in the *average.ipynb* notebook.


---

### Training a single model

To train a model, e.g. the base model (OpenRLHF/Llama-3-8b-sft-mixture) on task (variation) 1 and the first fold, run:

```bash
bash train.sh --model base --variation 1 --fold 1 # for the sft model (OpenRLHF/Llama-3-8b-sft-mixture)

bash train.sh --model instruct --variation 1 --fold 1 # for the rlhf model (OpenRLHF/OpenRLHF/Llama-3-8b-rlhf-100)

```

*There are also the a task '2' and 9 more folds.*


### Testing a sinlge task on a single fold

After training, testing the SFT and RLHF models for task (variation) 1 and the first fold can be done with the following command:

```bash
python test_model.py --variation 1 --fold 1
```
