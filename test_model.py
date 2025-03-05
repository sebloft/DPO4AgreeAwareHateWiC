import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import re


class ModelTester:
    def __init__(self, model, prompt_template, dataset_path, label_type="hate", temperature=0.001, max_new_tokens=5, answer_path="answers.json", cpu=False):

        if label_type not in ["hate", "quant"]:
            raise ValueError("Please use one of the following for label_type: hate, quant")
        self.label_type = label_type

        self.answer_path = answer_path

        # Sentence, Term
        self.prompt_template = prompt_template

        self.dataset_path = dataset_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')

        self.model_name = model
        
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.generation_params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            }

    def __del__(self):
        del self.model
        del self.tokenizer
        del self.device
        torch.cuda.empty_cache()

    def __str__(self):
        return f"""
        Tester on model {self.model_name}.

        Generation Params: {self.generation_params}
        """

    def get_test_data(self):
        """
        Returns the test data as list of tuples: (prompt, gold_label, gold_quantifier)
        """
        data_points = json.load(open(self.dataset_path, "r"))

        data = []

        for d in data_points:
            p = self.prompt_template.format(d["example"], d["term"])
            try:
                l = {"not hateful": 0, "hateful": 1, "undecided": 2}[d["majority_label"]]
            except KeyError:
                l = 2
            try:
                q = d["label2quantifier"]["hateful"]
                q = q.lower()
            except KeyError:
                q = "NaN"

            if q not in ["all", "most", "half", "few", "none"]:
                q = "NaN"

            data.append((p, l, q))

        return data


    def get_labels(self, text, len_prompt) -> tuple:
        """
        Returns tuple like (hate_label, quantifier_label)
        """

        text = text.lower().strip("<|end_of_text|>")

        if self.label_type == "hate":
            
            if re.findall(r'"not hateful"', text[len_prompt:]) or "does not express hate" in text[len_prompt:] or "does not express any hate" in text[len_prompt:]:
                return 0, None
            elif re.findall(r'"hateful"', text[len_prompt:]):
                return 1, None

            if "not hateful" in text[len_prompt:]:
                return 0, None
            elif "hateful" in text[len_prompt:]:
                return 1, None
            
            print("")
            print("NO LABEL FOUND IN:")
            print(text)
            print("----------------------------------------")
            print("")

            return 2, None


        elif self.label_type == "quant":
            quants = ["all", "most", "half", "few", "none"]
            quant2label = {
                "all": 1,
                "most": 1,
                "half": 2,
                "few": 0,
                "none": 0,
            }

            possible_answer = text[len_prompt:].strip()

            # finde "label"
            if match := re.findall(r'"(all|most|half|few|none)"', possible_answer):
                if len(set(match)) == 1:
                    q = match[0]
                    return quant2label[q], q

            # finde label
            if match := re.findall(r'(all|most|half|few|none)', possible_answer):
                if len(set(match)) == 1:
                    q = match[0]
                    return quant2label[q], q

            # finde ### response:
            #       "?label"?
            if match := re.findall(r'### response:\s*"?(all|most|half|few|none)"?\n', possible_answer):
                if len(set(match)) == 1 or len(re.findall(r'### response:\s*', text)) > 1:
                    q = match[0]
                    return quant2label[q], q

            # finde ### response:
            #       blabla "?label"? blabla
            if match := re.findall(r'### response:\s+.*"?(all|most|half|few|none)"?.*\n', possible_answer):
                if len(set(match)) == 1 or len(re.findall(r'### response:\s*', text)) > 1:
                    q = match[0]
                    return quant2label[q], q

            # finde response:
            #       "?label"?
            if match := re.findall(r'response:\s*"?(all|most|half|few|none)"?\n', possible_answer):
                if len(set(match)) == 1 or len(re.findall(r'### response:\s*', text)) > 1:
                    q = match[0]
                    return quant2label[q], q
            
            # finde response:
            #       blabla "?label"? blabla
            if match := re.findall(r'response:\s+.*"?(all|most|half|few|none)"?.*\n', possible_answer):
                if len(set(match)) == 1 or len(re.findall(r'### response:\s*', text)) > 1:
                    q = match[0]
                    return quant2label[q], q

            # finde sponse:
            #       "?label"?
            if match := re.findall(r'sponse:\s*"?(all|most|half|few|none)"?\n', possible_answer):
                if len(set(match)) == 1 or len(re.findall(r'### response:\s*', text)) > 1:
                    q = match[0]
                    return quant2label[q], q

            # finde sponse:
            #       blabla "?label"? blabla
            if match := re.findall(r'sponse:\s+.*"?(all|most|half|few|none)"?.*\n', possible_answer):
                if len(set(match)) == 1 or len(re.findall(r'### response:\s*', text)) > 1:
                    q = match[0]
                    return quant2label[q], q

            # finde: the quantifier "none" would be the most appropriate choice
            if match := re.findall(r'the quantifier "?(all|most|half|few|none)"? would be the most appropriate choice', possible_answer):
                if len(set(match)) == 1 or len(re.findall(r'### response:\s*', text)) > 1:
                    q = match[0]
                    return quant2label[q], q

            # finde 'that "q" .+ classify .+ as hateful'
            # if match := re.findall(r'that "?(all|most|half|few|none)"?.+classify.+as hateful', possible_answer):
            #     if len(set(match)) == 1 or len(re.findall(r'### response:\s*', text)) > 1:
            #         q = match[0]
            #         return quant2label[q], q


            if any(chosen_label := [l in possible_answer for l in quants]):
                if sum(chosen_label) == 1:
                    q = quants[chosen_label.index(True)]
                    return quant2label[q], q

            print("")
            print("NO QUANTIFIER FOUND IN:")
            print(text.strip())
            print("----------------------------------------")
            print("")
                    
            return 2, None

        raise NotImplementedError("Label type not 'hate' nor 'quant'")
            


    def prompt_model(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)

        output_ids = self.model.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id, **self.generation_params)
        output = self.tokenizer.decode(output_ids[0])
        
        return output


    def get_metrics(self, gold_labels, pred_labels, label_type="hate"):

        if label_type == "hate":
            labels = [0, 1, 2]
        elif label_type == "quant":
            labels = ["all", "most", "half", "few", "none", "NaN"]
        else:
            raise ValueError("Label type must be either 'hate' or 'quant'!!!")

        accuracy = accuracy_score(gold_labels, pred_labels)

        precision = precision_score(gold_labels, pred_labels, average="macro", zero_division=0)
        recall = recall_score(gold_labels, pred_labels, average="macro", zero_division=0)
        f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
        
        precision_pc = precision_score(gold_labels, pred_labels, average=None, labels=labels, zero_division=0)
        recall_pc = recall_score(gold_labels, pred_labels, average=None, labels=labels, zero_division=0)
        f1_pc = f1_score(gold_labels, pred_labels, average=None, labels=labels, zero_division=0)
        
        precision_text = ", ".join([str(labels[i]) + ": " + str(precision_pc[i]) for i in range(len(labels))])
        precision_text = f"{precision}\n\t{precision_text}"
        recall_text = ", ".join([str(labels[i]) + ": " + str(recall_pc[i]) for i in range(len(labels))])
        recall_text = f"{recall}\n\t{recall_text}"
        f1_text = ", ".join([str(labels[i]) + ": " + str(f1_pc[i]) for i in range(len(labels))])
        f1_text = f"{f1}\n\t{f1_text}"
        
        conf_matrix = confusion_matrix(gold_labels, pred_labels)

        return f"""

----------------------------------
Metrics for {self.model_name}{", Quantifier metrics" if label_type == "quant" else ""}:
----
Accuracy: {accuracy}
Precision: {precision_text}
Recall: {recall_text}
F1: {f1_text}
----
{conf_matrix}
----------------------------------

"""


    def test_model(self):
        data = self.get_test_data() # [(p, l, q)]

        gold_labels = []
        gold_quantifier = []

        pred_labels = []
        pred_quantifier = []

        if os.path.exists(self.answer_path):
            answers = json.load(open(self.answer_path))
        else:
            answers = dict()

        for i, (prompt, gold_l, gold_q) in enumerate(tqdm(data, desc="Prompting Model")):

            if gold_l == 2:
                continue

            if str((prompt, gold_l, gold_q)) in answers:
                answer = answers[str((prompt, gold_l, gold_q))]
            else:
                answer = self.prompt_model(prompt)
                answers[str((prompt, gold_l, gold_q))] = answer
            
            l, q = self.get_labels(answer, len(prompt))

            pred_labels.append(l)
            pred_quantifier.append(q if q else "NaN")

            gold_labels.append(gold_l)
            gold_quantifier.append(gold_q if gold_q else "NaN")

            if (i != 0 and i % 50 == 0) or i == len(data)-1:
                try:
                    print(f"\n\n___________________")
                    print(f"Evaluation at step {i}")
                    print(self.get_metrics(gold_labels, pred_labels, label_type="hate"))
                    print(self.get_metrics(gold_quantifier, pred_quantifier, label_type="quant"))
                    print("PRED LABELS: ", pred_labels)
                    print("PRED QUANTS: ", pred_quantifier)
                    print(f"-----------------------")
                except:
                    continue

            if (i != 0 and i % 10 == 0):
                json.dump(answers, open(self.answer_path, "w"))

        label_metrics = self.get_metrics(gold_labels, pred_labels, label_type="hate")
        quantifier_metrics = self.get_metrics(gold_quantifier, pred_quantifier, label_type="quant")

        json.dump(answers, open(self.answer_path, "w"))

        return label_metrics, gold_labels, pred_labels, quantifier_metrics, gold_quantifier, pred_quantifier


def eval_var_fold(variation=None, fold=1, cpu=False, models=["base", "OpenRLHF/Llama-3-8b-sft-mixture", "instruct", "OpenRLHF/Llama-3-8b-rlhf-100k"]):

    if variation == 1:
        prompt_template = """### Instruction:\nGiven the following sentence that mentions a particular term, classify whether the meaning of that term expresses hate towards a person or group within that specific sentence by giving one of the following corresponding labels:\n“hateful”\n“not hateful”\n\n### Input:\nSentence: "{}"\nTerm: "{}"\n### Response:"""
        label_type = "hate" 
        max_new_tokens = 50
    elif variation == 2:
        prompt_template = """### Instruction:\nGiven the following sentence that mentions a particular term, estimate how many of the human annotators would classify the meaning of that term as hateful by giving one of the following quantifiers:\n“all”\n“most”\n“half”\n“few”\n“none”\n\n### Input:\nSentence: "{}"\nTerm: "{}"\n\n### Response:\n"""
        label_type = "quant"
        max_new_tokens = 150
    else:
        raise NotImplementedError(f"Not implemented variation/task {variation}")

    for model in models:

        if model in ["base", "instruct"]:
            model = os.path.join(model, model + "_var" + str(variation) + "_fold" + str(fold))
            if not os.path.exists(model):
                raise FileExistsError(f"Path to model {model} not found!")

        dataset_path = os.path.join("data", "fold" + str(fold), "test.json")
        if not os.path.exists(dataset_path):
            raise FileExistsError(f"Path to dataset {dataset} not found!")

        file_ident = f"./eval/EVAL-TASK-{str(variation)}-FOLD-{str(fold)}--" + model.replace("/", "-").replace(" ", "_")

        tester = ModelTester(
            model=model,
            dataset_path=dataset_path,
            prompt_template=prompt_template,
            label_type=label_type,
            max_new_tokens=max_new_tokens,
            answer_path=file_ident + ".json",
            cpu=cpu,
            )

        print(tester)

        try:
            label_metrics, gold_labels, pred_labels, quantifier_metrics, gold_quantifier, pred_quantifier = tester.test_model()
        except:
            print(f"!!! PROBLEM WITH MODEL: {file_ident}")
            continue


        with open(file_ident + ".log", "w") as f:
            f.write(label_metrics)
            f.write("\nGOLD: " + str(gold_labels))
            f.write("\nPRED: " + str(pred_labels))

            f.write("\n\n\n-------------------------------")

            f.write(quantifier_metrics)
            f.write("\nGOLD QUANT: " + str(gold_quantifier))
            f.write("\nPRED QUANT: " + str(pred_quantifier))

        del tester


def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    parser.add_argument('--variation', type=int, required=True, help='The value of variation')
    parser.add_argument('--fold', type=int, required=True, help='The value of fold')
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        help='Only eval one model',
        default="",
        choices=["base", "OpenRLHF/Llama-3-8b-sft-mixture", "instruct", "OpenRLHF/Llama-3-8b-rlhf-100k"],
        )
    parser.add_argument('--cpu', type=bool, required=False, default=False, help='If no cuda should be used.')

    args = parser.parse_args()

    if not args.model:
        eval_var_fold(variation=args.variation, fold=args.fold, cpu=args.cpu)
    else:
        eval_var_fold(variation=args.variation, fold=args.fold, cpu=args.cpu, models=[args.model])
    
if __name__ == "__main__":
    main()
