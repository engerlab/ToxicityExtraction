import copy
from collections import defaultdict
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

def in_interval(span1,span2):
    if span1[0]>=span2[0] and span1[1]<=span2[1]:
        return True
    return False

class Sample:
    def __init__(self,fname,text,tokens,token_spans,tokens_str):
        self.token_ids=tokens
        self.fname=fname
        self.tokens=tokens_str
        self.token_spans=token_spans
        self.spans=dict()
        self.labels=[0]*len(tokens)
        self.text=text
        self.dose=list()
    def add_anno(self,id,type,span):
        self.spans[id]=[type,span,None]

    def add_labels(self):
        state=0
        for key in self.spans:
            span=self.spans[key]
            for indx,token_span in enumerate(self.token_spans):
                if state==0:
                    if in_interval(token_span,span[1]):
                        state=1
                        self.labels[indx]=1
                elif state==1:
                    if in_interval(token_span,span[1]):
                        self.labels[indx]=2
                    else:
                        state=0


def chunk_spans(elements):
    elements=copy.deepcopy(elements)
    elements.append(-1)
    state=0
    all=list()
    for indx,element in enumerate(elements):
        if state==0:
            if element==0 or element==2 or element==-1:
                continue
            else:
                current=list()
                current.append(indx)
                state=1
        elif state==1:
            if element==2:
                current.append(indx)
            elif element==0 or element==-1:
                all.append((current[0],current[-1]))
                state=0
            elif element==1:
                all.append((current[0], current[-1]))
                current=[indx]

    return all


def toxicity_duration(time_steps):
    first_one_index = next((i for i, tpl in enumerate(time_steps) if tpl[1] == 1), None)
    if first_one_index is None:
        return None,None

    remaining_tuples = [tpl for tpl in time_steps[first_one_index:]]

    converted_time=[(datetime.strptime(date[0],'%Y-%m-%d'),date[1]) for date in remaining_tuples]
    converted_time=[(date[0]-converted_time[0][0],date[1]) for date in converted_time]

    last_one_index = len(converted_time) - next((i for i, tpl in enumerate(reversed(converted_time)) if tpl[1] == 1), 1)-1
    if last_one_index==len(converted_time)-1:
        event=0
        dur=(converted_time[last_one_index][0].days/30,np.inf)
    else:
        if converted_time[last_one_index+1][0].days/30>20:
            aa=2
        dur=(converted_time[last_one_index][0].days/30,converted_time[last_one_index+1][0].days/30)
        event=1

    return dur,event


def evaluate(gold_spans, predicted_spans):
    """
    Evaluate NER predictions using strict and overlapping measures.

    Args:
        gold_spans (list of tuples): Gold standard spans, each represented as (start, end).
        predicted_spans (list of tuples): Predicted spans, each represented as (start, end).

    Returns:
        dict: A dictionary containing precision, recall, and F1 for strict and overlapping measures.
    """
    def overlap(span1, span2):
        """Check if two spans overlap."""
        return max(span1[0], span2[0]) < min(span1[1], span2[1])

    # Strict measure
    strict_matches = set(gold_spans) & set(predicted_spans)

    # Overlapping measure
    overlapping_matches = {
        pred for pred in predicted_spans
        for gold in gold_spans
        if overlap(pred, gold)
    }

    def calculate_metrics(matches, total_gold, total_pred):
        """Calculate precision, recall, and F1 score."""
        precision = len(matches) / total_pred if total_pred > 0 else 0
        recall = len(matches) / total_gold if total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1

    # Calculate metrics for strict and overlapping measures
    strict_metrics = calculate_metrics(strict_matches, len(gold_spans), len(predicted_spans))
    overlapping_metrics = calculate_metrics(overlapping_matches, len(gold_spans), len(predicted_spans))

    return {
        "strict": {"precision": strict_metrics[0], "recall": strict_metrics[1], "f1": strict_metrics[2]},
        "overlapping": {"precision": overlapping_metrics[0], "recall": overlapping_metrics[1], "f1": overlapping_metrics[2]},
    }
