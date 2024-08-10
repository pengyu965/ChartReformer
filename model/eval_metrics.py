import json
import pandas as pd
from copy import deepcopy
import pix2struct_metrics
from collections import Counter
import os 
from Chart_Reploter import Replot
from SSIM_eval import SSIM_score
from utils import resize_and_pad
from PIL import Image
"""Metrics functions for Chart Plotting Parameters by Jay Lal."""
def flatten_nested_dict(d, parent_key='', sep='-'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_nested_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            for index, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(flatten_nested_dict(item, f"{new_key}{sep}{index}", sep=sep))
                else:
                    items[f"{new_key}{sep}{index}"] = item
        else:
            items[new_key] = v
    return items

def compare_to_target(predicted_dict, target_dict):

    predicted_dict = flatten_nested_dict(predicted_dict.copy())
    target_dict = flatten_nested_dict(target_dict.copy())

    epsilon = 1e-5
    theta = 0.4 # anything above this is considered 100% error (no partial credit).
    key_scores = {}

    for key in target_dict:
        if key in predicted_dict:
            t_val, p_val = target_dict[key], predicted_dict[key]
            # Check if both values are numeric
            if isinstance(t_val, (int, float)) and isinstance(p_val, (int, float)):
                # Calculate continuous score for numeric values
                diff = min(abs(p_val - t_val) / abs(t_val + epsilon), 1)
                diff = 1 if diff > theta else diff
                score = 1 - diff
            else:
                # For non-numeric, (string or bool) use exact match
                score = 1 if t_val == p_val else 0
                # score = pix2struct_metrics.anls_metric(str(t_val),str(p_val),theta)
        else:
            # Key not present in predicted JSON
            score = 0

        key_scores[key] = round(score, 2)

    missing_keys = target_dict.keys() - predicted_dict.keys() 
    # penalize for predicted extra keys (by marking their score as 0)
    extra_keys = predicted_dict.keys() - target_dict.keys()

    # Calculate the similarity score
    total_score = 0 if not len(key_scores) else sum(key_scores.values())

    precision = total_score / len(predicted_dict) if len(predicted_dict) else 0
    recall = total_score / len(target_dict) if len(target_dict) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return_dict = {
        'precision': precision,
        'recall':recall,
        'f1': f1,
    }

    return return_dict


def eval_params_match(target_jsons, predicted_jsons):
    res = {
        'precision': [],
        'recall':[],
        'f1': []
    }
    
    for predicted_json, target_json in zip(predicted_jsons, target_jsons):
        param_wise_scores = compare_to_target(predicted_json, target_json)
        
        for key in res.keys():
            res[key].append(param_wise_scores[key])

    for key in res.keys():
        res[key] = 100 * sum(res[key])/len(target_jsons) if len(target_jsons) else 0
    
    return res

## ==================================================================================================
## ==================================================================================================

# Modified by Pengyu Yan
# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics functions for Chart and Table related tasks from Google Deplot."""

from collections.abc import Mapping, Sequence
import dataclasses
import itertools
from typing import Optional

import numpy as np
import pix2struct_metrics
from scipy import optimize


def _to_float(text):
    try:
        if text.endswith("%"):
            # Convert percentages to floats.
            return float(text.rstrip("%")) / 100.0
        else:
            return float(text)
    except ValueError:
        return None


def _get_relative_distance(
        target, prediction, theta = 1.0
):
    """Returns min(1, |target-prediction|/|target|)."""
    if not target:
        return int(not prediction)
    distance = min(abs((target - prediction) / target), 1)
    return distance if distance < theta else 1


def _table_numbers_match(target, prediction):
    """Calculates matching similarity between two tables following ChartQA."""

    target_numbers = _get_table_numbers(target)
    prediction_numbers = _get_table_numbers(prediction)
    if not target_numbers and not prediction_numbers:
        return 1
    if not target_numbers or not prediction_numbers:
        return 0
    max_len = max(len(target_numbers), len(prediction_numbers))
    distance = []
    for t in target_numbers:
        distance.append([_get_relative_distance(t, p) for p in prediction_numbers])
    cost_matrix = np.array(distance)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    return 1 - cost_matrix[row_ind, col_ind].sum() / max_len


def _get_table_numbers(text):
    numbers = []
    for line in text.splitlines():
        for part in line.split(" | "):
            if part.strip():
                try:
                    numbers.append(float(part))
                except ValueError:
                    pass
    return numbers


def table_number_accuracy_per_point(
        targets,
        predictions,
):
    """Calculates matching similarity between two tables following ChartQA.

    Keeps only numbers and performas a linear matching using the relative error.

    Args:
        targets: ground truth text.
        predictions: predicted text.

    Returns:
        A list of float numbers.
    """
    all_points_scores = []
    for p, targets in zip(predictions, targets):
        all_points_scores.append(max(_table_numbers_match(t, p) for t in targets))
    return all_points_scores


def table_number_accuracy(
        targets,
        predictions,
):
    """Aggregated version of table_number_accuracy_per_point().

    Same as table_number_accuracy_per_point() but returning an aggregated score.

    Args:
        targets: ground truth text.
        predictions: predicted text.

    Returns:
        dictionary with metric names as keys and metric value as values.
    """
    scores = table_number_accuracy_per_point(targets, predictions)
    return {"numbers_match": (100.0 * sum(scores)) / len(targets)}


def _permute(values, indexes):
    return tuple(values[i] if i < len(values) else "" for i in indexes)


@dataclasses.dataclass(frozen=True)
class Table:
    """Helper class for the content of a markdown table."""

    title: Optional[str] = None
    x_title: Optional[str] = None
    y_title: Optional[str] = None
    headers: tuple[str, Ellipsis] = dataclasses.field(default_factory=tuple)
    rows: tuple[tuple[str, Ellipsis], Ellipsis] = dataclasses.field(default_factory=tuple)

    def permuted(self, indexes):
        """Builds a version of the table changing the column order."""
        return Table(
                title=self.title,
                x_title=self.x_title,
                y_title=self.y_title,
                headers=_permute(self.headers, indexes),
                rows=tuple(_permute(row, indexes) for row in self.rows),
        )

    def aligned(
            self, headers, text_theta = 0.5
    ):
        """Builds a column permutation with headers in the most correct order."""
        if len(headers) != len(self.headers):
            raise ValueError(f"Header length {headers} must match {self.headers}.")
        distance = []
        for h2 in self.headers:
            distance.append(
                    [
                            1 - pix2struct_metrics.anls_metric(h1, h2, text_theta)
                            for h1 in headers
                    ]
            )
        cost_matrix = np.array(distance)
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        permutation = [idx for _, idx in sorted(zip(col_ind, row_ind))]
        score = (1 - cost_matrix)[permutation[1:], range(1, len(row_ind))].prod()
        return self.permuted(permutation), score


def _parse_table(table, transposed = False):
    """Builds a table from a markdown representation."""

    text = table['data_table'] if 'data_table' in table.keys() else None
    title = table['chart_title'] if 'chart_title' in table.keys() else None
    x_title = table['x_axis_title'] if 'x_axis_title' in table.keys() else None
    y_title = table['y_axis_title'] if 'y_axis_title' in table.keys() else None

    lines = text.lower().splitlines()
    if not lines:
        return Table(title=title, x_title=x_title, y_title=y_title)

    rows = []
    for line in lines:
        rows.append(tuple(v.strip() for v in line.split(" | ")))
    if transposed:
        rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]

    return Table(title=title, x_title=x_title, y_title=y_title, headers=rows[0], rows=tuple(rows[1:]))


def _get_table_datapoints(table):
    """Extracts a dict of datapoints from a table."""
    datapoints = {}
    if table.title is not None:
        datapoints["title"] = table.title
    if table.x_title is not None:
        datapoints["x_title"] = table.x_title
    if table.y_title is not None:
        datapoints["y_title"] = table.y_title
    if not table.rows or len(table.headers) <= 1:
        return datapoints
    for row in table.rows:
        for header, cell in zip(table.headers[1:], row[1:]):
            datapoints[f"{row[0]} {header}"] = cell
    return datapoints


def _get_datapoint_metric(
        target,
        prediction,
        text_theta=0.5,
        number_theta=0.1,
):
    """Computes a metric that scores how similar two datapoint pairs are."""
    key_metric = pix2struct_metrics.anls_metric(
            target[0], prediction[0], text_theta
    )
    pred_float = _to_float(prediction[1])
    target_float = _to_float(target[1])
    if pred_float is not None and target_float:
        return key_metric * (
                1 - _get_relative_distance(target_float, pred_float, number_theta)
        )
    elif target[1] == prediction[1]:
        return key_metric
    else:
        return key_metric * pix2struct_metrics.anls_metric(
                target[1], prediction[1], text_theta
        )


def _table_datapoints_precision_recall_f1(
        target_table,
        prediction_table,
        text_theta = 0.5,
        number_theta = 0.1,
):
    """Calculates matching similarity between two tables as dicts."""
    target_datapoints = list(_get_table_datapoints(target_table).items())
    prediction_datapoints = list(_get_table_datapoints(prediction_table).items())
    if not target_datapoints and not prediction_datapoints:
        return 1, 1, 1
    if not target_datapoints:
        return 0, 1, 0
    if not prediction_datapoints:
        return 1, 0, 0
    distance = []
    for t, _ in target_datapoints:
        distance.append(
                [
                        1 - pix2struct_metrics.anls_metric(t, p, text_theta)
                        for p, _ in prediction_datapoints
                ]
        )
    cost_matrix = np.array(distance)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    score = 0
    for r, c in zip(row_ind, col_ind):
        score += _get_datapoint_metric(
                target_datapoints[r], prediction_datapoints[c], text_theta, number_theta
        )
    if score == 0:
        return 0, 0, 0
    precision = score / len(prediction_datapoints)
    recall = score / len(target_datapoints)
    return precision, recall, 2 * precision * recall / (precision + recall)

def table_datapoints_precision_recall_per_point(
        targets,
        predictions,
        text_theta = 0.5,
        number_theta = 0.1,
):
    """Computes precisin recall and F1 metrics given two flattened tables.

    Parses each string into a dictionary of keys and values using row and column
    headers. Then we match keys between the two dicts as long as their relative
    levenshtein distance is below a threshold. Values are also compared with
    ANLS if strings or relative distance if they are numeric.

    Args:
        targets: list of list of strings.
        predictions: list of strings.
        text_theta: relative edit distance above this is set to the maximum of 1.
        number_theta: relative error rate above this is set to the maximum of 1.

    Returns:
        Dictionary with per-point precision, recall and F1
    """
    assert len(targets) == len(predictions)
    per_point_scores = {"precision": [], "recall": [], "f1": []}
    for pred, target in zip(predictions, targets):
        all_metrics = []
        for transposed in [True, False]:
            pred_table = _parse_table(pred, transposed=transposed)
            # pylint:disable=g-complex-comprehension
            all_metrics.extend(
                    [
                            _table_datapoints_precision_recall_f1(
                                    _parse_table(t),
                                    pred_table,
                                    text_theta,
                                    number_theta,
                            )
                            for t in target
                    ]
            )
            # pylint:enable=g-complex-comprehension
        p, r, f = max(all_metrics, key=lambda x: x[-1])
        per_point_scores["precision"].append(p)
        per_point_scores["recall"].append(r)
        per_point_scores["f1"].append(f)
    return per_point_scores


def table_datapoints_precision_recall(
        targets,
        predictions,
        text_theta = 0.5,
        number_theta = 0.1,
):
    """Aggregated version of table_datapoints_precision_recall_per_point().

    Same as table_datapoints_precision_recall_per_point() but returning aggregated
    scores instead of per-point scores.

    Args:
        targets: list of list of strings.
        predictions: list of strings.
        text_theta: relative edit distance above this is set to the maximum of 1.
        number_theta: relative error rate above this is set to the maximum of 1.

    Returns:
        Dictionary with aggregated precision, recall and F1
    """
    score_dict = table_datapoints_precision_recall_per_point(
            targets, predictions, text_theta, number_theta
    )
    return {
            "table_datapoints_precision": (
                    100.0 * sum(score_dict["precision"]) / len(targets)
            ),
            "table_datapoints_recall": (
                    100.0 * sum(score_dict["recall"]) / len(targets)
            ),
            "table_datapoints_f1": 100.0 * sum(score_dict["f1"]) / len(targets),
    }


def _get_row_datapoints(table):
    """Extracts a list of datapoints from a table as rows."""
    if table.title is None:
        return table.rows
    return table.rows + (("title", table.title),("x_title", table.x_title),("y_title", table.y_title))


def _get_row_metric(
        target_parts,
        prediction_parts,
        text_theta=0.5,
        number_theta=0.1,
):
    """Computes a metric that scores how similar two datapoint pairs are."""
    if len(target_parts) != len(prediction_parts) or not target_parts:
        return 0.0
    result = []
    for target, prediction in zip(target_parts, prediction_parts):
        pred_float = _to_float(prediction)
        target_float = _to_float(target)
        if target == prediction:
            result.append(1.0)
        elif pred_float is not None and target_float:
            result.append(
                    1 - _get_relative_distance(target_float, pred_float, number_theta)
            )
        elif target_float is not None:
            result.append(0.0)
        else:
            result.append(
                    pix2struct_metrics.anls_metric(target, prediction, text_theta)
            )
    return np.prod(result)


def _row_datapoints_precision_recall_f1(
        target,
        prediction,
        text_theta = 0.5,
        number_theta = 0.1,
):
    """Calculates matching similarity between two tables as list of rows."""
    target_datapoints = _get_row_datapoints(target)
    aligned_prediction, aligned_score = prediction.aligned(
            target.headers, text_theta
    )
    prediction_datapoints = _get_row_datapoints(aligned_prediction)
    if not target_datapoints and not prediction_datapoints:
        return 1, 1, 1
    if not target_datapoints:
        return 0, 1, 0
    if not prediction_datapoints or not aligned_score:
        return 1, 0, 0
    metrics = []
    for t in target_datapoints:
        metrics.append(
                [
                        aligned_score * _get_row_metric(t, p, text_theta, number_theta)
                        for p in prediction_datapoints
                ]
        )
    metrics_matrix = np.array(metrics)
    row_ind, col_ind = optimize.linear_sum_assignment(1 - metrics_matrix)
    score = metrics_matrix[row_ind, col_ind].sum()
    if score == 0:
        return 0, 0, 0
    precision = score / len(prediction_datapoints)
    recall = score / len(target_datapoints)
    return precision, recall, 2 * precision * recall / (precision + recall)


def row_datapoints_precision_recall(
        targets,
        predictions,
        text_theta = 0.5,
        number_theta = 0.1,
):
    """Computes precisin recall and F1 metrics given two flattened tables.

    Parses each string into a list of rows using column headers. Then we match
    entries by their levenshtein / numeric relative distance is below a threshold.

    Args:
        targets: list of list of strings.
        predictions: list of strings.
        text_theta: relative edit distance above this is set to the maximum of 1.
        number_theta: relative error rate above this is set to the maximum of 1.

    Returns:
        Mapping with precision, recall and F1
    """
    if len(targets) != len(predictions):
        raise ValueError(
                f"Targets has length {len(targets)} and predictions has length "
                f"{len(predictions)}."
        )
    precision, recall, f1 = 0, 0, 0
    for pred, target in zip(predictions, targets):
        all_metrics = []
        prediction_tables = [
                _parse_table(pred, transposed=transposed)
                for transposed in [True, False]
        ]
        for t in target:
            for target_transposed in [True, False]:
                target_table = _parse_table(t, transposed=target_transposed)
                for prediction_table in prediction_tables:
                    if len(target_table.headers) != len(prediction_table.headers):
                        continue
                    all_metrics.append(
                            _row_datapoints_precision_recall_f1(
                                    target_table,
                                    prediction_table,
                                    text_theta,
                                    number_theta,
                            )
                    )
        p, r, f = max(all_metrics, key=lambda x: x[-1], default=(0, 0, 0))
        precision += p
        recall += r
        f1 += f
    return {
        "row_datapoints_precision": 100.0 * precision / len(targets),
        "row_datapoints_recall": 100.0 * recall / len(targets),
        "row_datapoints_f1": 100.0 * f1 / len(targets),
    }

def process_param_json(json_f, 
                       pop_keys=['data_table','chart_title','x_axis_title','y_axis_title']):
    table = {}
    params = deepcopy(json_f)
    for key in pop_keys:
        if key in params.keys():
            table[key] = params.pop(key)

    if 'data_table' in table.keys():
        table['data_table'] = table['data_table'].replace('<0x0A>', '\n')
    else:
        table['data_table'] = ""

    return table, params

def eval_pretraining():
    import json
    import os
    from tqdm import tqdm
    import json_repair
    import ast

    gt_path = '../../data/finetune_final_sep/test/edited_parameters/'
    pre_path = './results/pretraining_laterwork/pre_on_edited_parameters/'

    gt_table_list = []
    pre_table_list = []
    
    gt_param_list = []
    pre_param_list = []


    unable_read_pre_list = []
    for file in tqdm(os.listdir(pre_path)):
        with open(gt_path + file, 'r') as f:
            target_json = json.load(f)

        try:
            with open(pre_path + file, 'r') as f:
                predicted_json = json_repair.loads(f.read())
                # break
        except:
            unable_read_pre_list.append(file)
            continue
        
        # try:
        #     with open(pre_path + file, 'r') as f:
        #         predicted_json = json.load(f)
        # except:
        #     unable_read_pre_list.append(file)
        #     # with open(pre_path + file, 'r') as f:
        #     #     predicted_json = json_repair.loads(f.read())
        #     # predicted_table, predicted_params = process_param_json(predicted_json, repaired=True)
        #     # print(predicted_params)
        #     # print(predicted_table)
        #     # break
        #     continue

        target_table, target_params = process_param_json(target_json, repaired=False)
        predicted_table, predicted_params = process_param_json(predicted_json, repaired=True)
        # print(target_table["underlying_data"])
        # print(predicted_table["underlying_data"])
        # predicted_table['x_axis_title'] = "abcdef"
        # print(predicted_params)
        # print(row_datapoints_precision_recall([[target_table]], [predicted_table]))
        # print(table_datapoints_precision_recall([[target_table]], [predicted_table]))

        gt_table_list.append([target_table])
        pre_table_list.append(predicted_table)

        gt_param_list.append(target_params)
        pre_param_list.append(predicted_params)

        # break

    print("Not intact json file predicted:",unable_read_pre_list)
    print("Number of intact json file predicted:",len(unable_read_pre_list))

    print(len(gt_table_list), len(pre_table_list))
    print(row_datapoints_precision_recall(gt_table_list, pre_table_list))
    print(table_datapoints_precision_recall(gt_table_list, pre_table_list))
    print(eval_params_match(gt_param_list, pre_param_list))

    count = 0
    for gt_param, pre_param in zip(gt_param_list, pre_param_list):
        if gt_param == pre_param:
            count += 1

    print("total match parameters count:{}/{}".format(count, len(gt_param_list)))

def eval_chartediting():
    import json
    import os
    from tqdm import tqdm
    import json_repair
    import ast

    def compare_json_dicts(ori_dict, edit_dict):
        """
        Compare two dictionaries and return a list of keys for which the values have changed.

        :param dict1: First dictionary to compare.
        :param dict2: Second dictionary to compare.
        :return: List of keys where the values have changed.
        """
        changed_keys = []

        # Check all keys in the first dictionary
        for key in edit_dict:
            if key in ori_dict:
                if edit_dict[key] != ori_dict[key]:
                    changed_keys.append(key)
            else:
                # Key exists in dict1 but not in dict2; treat this as a change
                changed_keys.append(key)
        
        # Check for any keys that are in dict2 but not in dict1

        return changed_keys

    gt_folder = '../../data/new_edits_data/test_small/'
    gt_summary = json.load(open(gt_folder + '/summary.json','r'))

    pre_path = './results/finetune_new_edits_data_small/'

    cates_eval_dict = {}
    total_eval_dict = {
                "data_table_metrics":{
                    "precision":[],
                    "recall":[],
                    "f1":[]
                },
                "visual_attr_metrics":{
                    "precision":[],
                    "recall":[],
                    "f1":[]
                },
                "ssim":[],
                "success_rate":[]
            }

    for idx in tqdm(gt_summary.keys()):
        gt_item = gt_summary[idx]

        edit_type = gt_item['edit_type']
        original_vattr = gt_item['visual_attributes']
        gt_vattr = gt_item['edited_visual_attributes']
        gt_data_table = gt_item['edited_underlying_data']
        gt_data_table['data_table'] = gt_data_table['data_table'].replace('<0x0A>', '\n')

        chart_id = os.path.splitext(os.path.split(gt_item['original_image'])[-1])[0]
        try:
            with open(pre_path+chart_id+".json", 'r') as f:
                pre_json = json_repair.loads(f.read())
                # break
        except:
            # unable_read_pre_list.append(file)
            continue

        pre_data_table, pre_vattr = process_param_json(pre_json)

        original_vattr = flatten_nested_dict(original_vattr)
        gt_vattr = flatten_nested_dict(gt_vattr)
        changed_key = compare_json_dicts(original_vattr, gt_vattr)

        if "format_edits" in edit_type:
            type_related_changed_key = [key for key in changed_key if 'chart_type' in key]
            changed_key = [key for key in changed_key if 'color' in key] + type_related_changed_key

        pre_vattr = flatten_nested_dict(pre_vattr)

        gt_vattr_noedit_dict, gt_vattr_edit_dict, pre_vattr_noedit_dict, pre_vattr_edit_dict = {},{},{},{}

        for key in gt_vattr.keys():
            if key in changed_key:
                gt_vattr_edit_dict[key] = gt_vattr[key]
            else:
                gt_vattr_noedit_dict[key] = gt_vattr[key]

        for key in pre_vattr.keys():
            if key in changed_key:
                pre_vattr_edit_dict[key] = pre_vattr[key]
            else:
                pre_vattr_noedit_dict[key] = pre_vattr[key]

        if changed_key == []:
            gt_vattr_edit_dict["gold"] = "gold"
            pre_vattr_edit_dict["gold"] = "gold"

        data_table_metrics = table_datapoints_precision_recall([[gt_data_table]], [pre_data_table])

        shouldnot_edit_eval = eval_params_match([gt_vattr_noedit_dict], [pre_vattr_noedit_dict])
        should_edit_eval = eval_params_match([gt_vattr_edit_dict], [pre_vattr_edit_dict])
        visual_attr_metrics = {}
        for key in should_edit_eval.keys():
            visual_attr_metrics[key] = (should_edit_eval[key] + shouldnot_edit_eval[key])/2
            
        if edit_type[1] not in cates_eval_dict.keys():
            cates_eval_dict[edit_type[1]] = {
                "data_table_metrics":{
                    "precision":[],
                    "recall":[],
                    "f1":[]
                },
                "visual_attr_metrics":{
                    "precision":[],
                    "recall":[],
                    "f1":[]
                },
                "ssim":[],
                "success_rate":[]
            }

        for key in ["precision","recall","f1"]:
            cates_eval_dict[edit_type[1]]["data_table_metrics"][key].append(data_table_metrics["table_datapoints_"+key])
            cates_eval_dict[edit_type[1]]["visual_attr_metrics"][key].append(visual_attr_metrics[key])
            total_eval_dict["data_table_metrics"][key].append(data_table_metrics["table_datapoints_"+key])
            total_eval_dict["visual_attr_metrics"][key].append(visual_attr_metrics[key])


        try:
            replot = Replot(pre_json)
            replot.fig.savefig(pre_path + chart_id + "_replot.png")
            cates_eval_dict[edit_type[1]]["success_rate"].append(1)
            total_eval_dict["success_rate"].append(1)

            img1 = resize_and_pad(Image.open(gt_folder + "edited_images/" + chart_id + ".png").convert('RGB'),(800,800))
            img2 = resize_and_pad(Image.open(pre_path + chart_id + "_replot.png").convert('RGB'),(800,800))
            ssim = SSIM_score(np.array(img1), np.array(img2))
            print(ssim)

            cates_eval_dict[edit_type[1]]["ssim"].append(ssim)
            total_eval_dict["ssim"].append(ssim)
        except:
            cates_eval_dict[edit_type[1]]["success_rate"].append(0)
            total_eval_dict["success_rate"].append(0)
            pass

    for cate in cates_eval_dict.keys():
        print("Evaluation for", cate, "=========================================")
        print("Metric for data table:", {
            key: (sum(cates_eval_dict[cate]["data_table_metrics"][key]) / len(cates_eval_dict[cate]["data_table_metrics"][key])) for key in cates_eval_dict[cate]["data_table_metrics"].keys()
        })
        print("Metric for visual attributes:", {
            key: (sum(cates_eval_dict[cate]["visual_attr_metrics"][key]) / len(cates_eval_dict[cate]["visual_attr_metrics"][key])) for key in cates_eval_dict[cate]["visual_attr_metrics"].keys()
        }) 
        print("SSIM:", sum(cates_eval_dict[cate]["ssim"]) / len(cates_eval_dict[cate]["ssim"]))
        print("Success rate:", sum(cates_eval_dict[cate]["success_rate"]) / len(cates_eval_dict[cate]["success_rate"]))

    print("Total Evaluation ==========================================")
    print("Metric for data table:", {
        key: (sum(total_eval_dict["data_table_metrics"][key]) / len(total_eval_dict["data_table_metrics"][key])) for key in total_eval_dict["data_table_metrics"].keys()
    })
    print("Metric for visual attributes:", {
        key: (sum(total_eval_dict["visual_attr_metrics"][key]) / len(total_eval_dict["visual_attr_metrics"][key])) for key in total_eval_dict["visual_attr_metrics"].keys()
    }) 
    print("SSIM:", sum(total_eval_dict["ssim"]) / len(total_eval_dict["ssim"]))
    print("Success rate:", sum(total_eval_dict["success_rate"]) / len(total_eval_dict["success_rate"]))


if __name__ == "__main__":
    # eval_pretraining()
    eval_chartediting()
