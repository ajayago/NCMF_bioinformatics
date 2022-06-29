import os
from typing import Union

import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
#from sklearn.utils.fixes import signature
from funcsigs import signature


def get_auroc(group: pd.DataFrame) -> pd.Series:
    """
    :param group: pd.DataFrame with predictions. Should have 2 columns: truth (0 or 1 ground truth label) and result (float prediction)
    :return: pd.Series with auroc, auprc, and ap50
    """
    fpr, tpr, thresholds = roc_curve(group.truth, group.result, pos_label=1)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(group.truth, group.result)
    ap50 = apk(group.truth, group.result)
    return pd.Series({"auroc": roc_auc, "auprc": pr_auc, "ap50": ap50})

def get_aurocs_by_side_effect(positive_examples: pd.DataFrame, negative_examples: Union[pd.DataFrame, list],
                              positive_results: Union[pd.DataFrame, list], negative_results: pd.DataFrame,
                              pr_curve: bool = False,
                              plot: bool = True, single_side_effect: str = None, verbose=True,
                              round_to_digits=3) -> pd.DataFrame:
    """
    :param positive_examples: pd.DataFrame with positive testing samples
    :param negative_examples: pd.DataFrame with negative testing samples
    :param positive_results: pd.DataFrame with positive predictions
    :param negative_results: pd.DataFrame with negative predictions
    :param pr_curve: if true, plot the precision recall curve
    :param plot: if true, plot the receiver operating characteristic
    :param single_side_effect: if true, report details for this particular side effect
    :param verbose: print results, including mean/median and confidence intervals, rather than just returning the result dataframe.
    :param round_to_digits: round the output to this number of digits (default 3)
    :return: pd.DataFrame with aurocs for each side effect, as well as the number of positive and negative samples for each side effect (for sanity check)
    """
    # ensure correct format and columns names
    if type(negative_results) is not pd.DataFrame:
        negative_results = pd.DataFrame(negative_results)
    if type(positive_results) is not pd.DataFrame:
        positive_results = pd.DataFrame(positive_results)

    negative_results.columns = [0]
    positive_results.columns = [0]
    if 'predicate_name' not in positive_examples.columns and 'predicate' in positive_examples.columns:
        rename_dict = {'predicate': 'predicate_name', 'subject': 'subject_name', 'object': 'object_name'}
        positive_examples.rename(columns=rename_dict, inplace=True)
        negative_examples.rename(columns=rename_dict, inplace=True)

    if verbose:
        if single_side_effect is not None:
            print("Reporting on", single_side_effect, "only.")

        print("Positive results mean score:", round(positive_results[0].mean(), round_to_digits))
        print("Negative results mean score:", round(negative_results[0].mean(), round_to_digits))

    negative_examples, positive_examples = update_examples_for_single_side_effect(negative_examples, positive_examples,
                                                                                  single_side_effect)

    all_examples = create_all_examples_df(negative_examples, negative_results, positive_examples, positive_results,
                                          single_side_effect)

    ap50, fpr, pr_auc, roc_auc, tpr = calculate_metrics(all_examples)

    if verbose:
        print("Overall AUROC:", round(roc_auc, round_to_digits))
        print("Overall AUPRC:", round(pr_auc, round_to_digits))
        print("Overall AP50:", round(ap50, round_to_digits))

    # calculate aurocs per side effect.
    aurocs_by_side_effect = all_examples.groupby(by='predicate_name').apply(get_auroc)
    print(aurocs_by_side_effect.head)
    if verbose:
        metrics = pd.DataFrame(columns=["auroc", "auprc", "ap50"])
        metrics = metrics.append(pd.Series(
            {"auroc": aurocs_by_side_effect.auroc.median(), "auprc": aurocs_by_side_effect.auprc.median(),
             "ap50": aurocs_by_side_effect.ap50.median()}, name="median"))
        metrics = metrics.append(pd.Series(
            {"auroc": aurocs_by_side_effect.auroc.mean(), "auprc": aurocs_by_side_effect.auprc.mean(),
             "ap50": aurocs_by_side_effect.ap50.mean()}, name="mean"))
        metrics = metrics.append(
            pd.Series({"auroc": aurocs_by_side_effect.auroc.std(), "auprc": aurocs_by_side_effect.auprc.std(),
                       "ap50": aurocs_by_side_effect.ap50.std()}, name="std"))
        metrics = metrics.append(
            pd.Series({"auroc": aurocs_by_side_effect.auroc.sem(), "auprc": aurocs_by_side_effect.auprc.sem(),
                       "ap50": aurocs_by_side_effect.ap50.sem()}, name="sem"))
        metrics = metrics.append(pd.Series(
            {"auroc": aurocs_by_side_effect.auroc.min(), "auprc": aurocs_by_side_effect.auprc.min(),
             "ap50": aurocs_by_side_effect.ap50.min()}, name="min"))
        metrics = metrics.append(pd.Series(
            {"auroc": aurocs_by_side_effect.auroc.max(), "auprc": aurocs_by_side_effect.auprc.max(),
             "ap50": aurocs_by_side_effect.ap50.max()}, name="max"))
        metrics = metrics.transpose()
        print()
        print("Average performance over {} side effects:".format(len(aurocs_by_side_effect)))
        print(metrics.round(round_to_digits))

    if plot:
        import matplotlib.pyplot as plt
        # plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.show()

    if pr_curve:
        # plot PR curve
        make_precision_recall_curve(all_examples)

    # sanity check: find number of positive and negative examples per side effect
    df = pd.DataFrame(all_examples.groupby(
        by=['predicate_name', 'truth']).subject_name.count()).reset_index()
    df = df.pivot(index='predicate_name', columns='truth')

    sample_sizes = pd.DataFrame(
        {'negative_examples': df['subject_name'][0], 'positive_examples': df['subject_name'][1]})
    return aurocs_by_side_effect.join(sample_sizes)


def calculate_metrics(all_examples: pd.DataFrame) -> (float, float, float, float, float):
    """
    :param all_examples: pd.DataFrame of ground truth labels (in column "truth") and predictions (in column "result")
    :return: tuple with: average precision at 50, false positive rate, area under the precision recall curve, area under the receiver operating characteristic, true positive rate
    """
    fpr, tpr, thresholds = roc_curve(all_examples.truth, all_examples.result, pos_label=1)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(all_examples.truth, all_examples.result)
    ap50 = apk(all_examples.truth, all_examples.result)
    return ap50, fpr, pr_auc, roc_auc, tpr


def create_all_examples_df(negative_examples, negative_results, positive_examples, positive_results,
                           single_side_effect):
    """put positive and negative examples in one data frame"""
    negative_examples = negative_examples.copy()
    positive_examples = positive_examples.copy()
    if "result" in negative_examples.columns:
        negative_examples.drop("result", axis=1, inplace=True)
    negative_examples.insert(0, "result", negative_results[0])
    if "result" in positive_examples.columns:
        positive_examples.drop("result", axis=1, inplace=True)
    positive_examples.insert(0, "result", positive_results[0])
    if "truth" in negative_examples.columns:
        negative_examples.drop("truth", axis=1, inplace=True)
    negative_examples.insert(0, "truth", 0)
    if "truth" in positive_examples.columns:
        positive_examples.drop("truth", axis=1, inplace=True)
    positive_examples.insert(0, "truth", 1)
    # put positive and negative examples in one data frame
    all_examples = positive_examples.append(negative_examples)
    return all_examples


def update_examples_for_single_side_effect(negative_examples: pd.DataFrame, positive_examples: pd.DataFrame,
                                           single_side_effect: str = None) -> (pd.DataFrame, pd.DataFrame):
    """if we are looking at a single side effect, remove everything else"""
    if single_side_effect is not None:
        if "predicate_name" in positive_examples.columns:
            positive_examples = positive_examples[positive_examples.predicate_name == single_side_effect]
            negative_examples = negative_examples[negative_examples.predicate_name == single_side_effect]
        else:
            positive_examples = positive_examples[positive_examples.predicate == single_side_effect]
            negative_examples = negative_examples[negative_examples.predicate == single_side_effect]
    return negative_examples, positive_examples


def get_aurocs_by_side_effect_files(file_location, positive_examples_file_name="positive_examples.tsv",
                                    negative_examples_file_name="negative_examples.tsv",
                                    positive_scores_file_name="positive_examples_scores.txt",
                                    negative_scores_file_name="negative_examples_scores.txt", plot=True,
                                    single_side_effect=None) -> pd.DataFrame:
    """
    :param file_location: path where positive_examples.tsv, negative_examples.tsv, positive_examples_scores_i.txt, and negative_examples_scores_i.txt are located. tsv files should contain tab-separated subject-predicate-object triples.
    :param positive_examples_file_name: tab separated file with positive examples
    :param negative_examples_file_name: tab separated file with negative examples
    :param positive_scores_file_name: txt file with the list of scores for positive examples
    :param negative_scores_file_name: txt file with the list of scores for negative examples
    :param plot: if true, plot ROC curve
    :param single_side_effect: if given, focus on this side effect
    :return: pd.DataFrame with aurocs for each side effect, as well as the number of positive and negative samples for each side effect (for sanity check)
    """
    positive_examples = pd.read_csv(
        file_location + positive_examples_file_name, sep='\t', header=None)
    negative_examples = pd.read_csv(
        file_location + negative_examples_file_name, sep='\t', header=None)
    positive_examples.columns = ['subject_name', 'predicate_name', 'object_name']
    negative_examples.columns = ['subject_name', 'predicate_name', 'object_name']
    neg_scores = pd.read_csv(
        file_location + negative_scores_file_name, header=None)
    pos_scores = pd.read_csv(
        file_location + positive_scores_file_name, header=None)
    return get_aurocs_by_side_effect(positive_examples=positive_examples, negative_examples=negative_examples,
                                     positive_results=pos_scores, negative_results=neg_scores, plot=plot,
                                     single_side_effect=single_side_effect)


def make_precision_recall_curve(test_results_df: pd.DataFrame):
    """
    Assumes columns result and truth in the input dataframe. result should have the score, and truth the actual label
    (0 or 1). Returns nothing
    """
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(test_results_df.truth, test_results_df.result)
    pr_auc = average_precision_score(test_results_df.truth, test_results_df.result)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(pr_auc))


def add_curve(positive_examples: pd.DataFrame, negative_examples: pd.DataFrame, positive_results: pd.DataFrame,
              negative_results: pd.DataFrame, iteration: int, metrics: pd.DataFrame, plot: bool = True,
              single_side_effect: str = None) -> pd.DataFrame:
    negative_examples, positive_examples = update_examples_for_single_side_effect(negative_examples, positive_examples,
                                                                                  single_side_effect)

    all_examples = create_all_examples_df(negative_examples, negative_results, positive_examples, positive_results,
                                          single_side_effect)

    ap50, fpr, pr_auc, roc_auc, tpr = calculate_metrics(all_examples)

    aurocs_by_se = get_aurocs_by_side_effect(positive_examples, negative_examples, positive_results, negative_results,
                                             False, False, single_side_effect, verbose=False)
    mean_auroc = aurocs_by_se.auroc.mean()
    auroc_min = aurocs_by_se.auroc.min()
    auroc_max = aurocs_by_se.auroc.max()
    mean_auprc = aurocs_by_se.auprc.mean()
    auprc_min = aurocs_by_se.auprc.min()
    auprc_max = aurocs_by_se.auprc.max()
    mean_ap50 = aurocs_by_se.ap50.mean()
    ap50_min = aurocs_by_se.ap50.min()
    ap50_max = aurocs_by_se.ap50.max()

    metrics = metrics.append(pd.DataFrame(
        {"mean_auprc": mean_auprc, "mean_auroc": mean_auroc, "mean_ap50": mean_ap50,
         "auroc_min": auroc_min, "auroc_max": auroc_max,
         "auprc_min": auprc_min, "auprc_max": auprc_max,
         "ap50_min": ap50_min, "ap50_max": ap50_max,
         },
        index=[iteration]))

    if plot:
        import matplotlib.pyplot as plt
        # plot ROC curve
        plt.plot(fpr, tpr, label='%d (area = %0.3f)' % (iteration, roc_auc))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.show()
    return metrics


def truncation_analysis(file_location: str, truncation_dimensions: list = [16000, 8000, 4032, 1024, 512, 256, 128, 64],
                        plot: bool = True, single_side_effect: str = None) -> pd.DataFrame:
    """
    :param single_side_effect:
    :param file_location: path where positive_examples.tsv, negative_examples.tsv, positive_examples_scores_i.txt, and negative_examples_scores_i.txt are located. tsv files should contain tab-separate subject-predicate-object triples.
    :param truncation_dimensions: list of dimensions for which there are score files in the file_location.
    :param plot: True if a plot should be created; false otherwise.
    :return: data frame that lists auroc, auprc, ap50, and mean positive and negative similarity scores per dimensionality
    """
    metrics = pd.DataFrame()

    positive_examples = pd.read_csv(file_location + "positive_examples.tsv", sep='\t', header=None)
    negative_examples = pd.read_csv(file_location + "negative_examples.tsv", sep='\t', header=None)

    positive_examples.columns = ['subject_name', 'predicate_name', 'object_name']
    negative_examples.columns = ['subject_name', 'predicate_name', 'object_name']

    for i in truncation_dimensions:
        neg_scores = pd.read_csv(
            file_location + "negative_examples_scores_" + str(i) + ".txt", header=None)
        pos_scores = pd.read_csv(
            file_location + "positive_examples_scores_" + str(i) + ".txt", header=None)
        metrics = add_curve(
            positive_examples=positive_examples, negative_examples=negative_examples, positive_results=pos_scores,
            negative_results=neg_scores, iteration=i,
            metrics=metrics, plot=False, single_side_effect=single_side_effect)

    if plot:
        import matplotlib.pyplot as plt
        ax = metrics.plot(y="mean_auroc",
                          grid=True, legend=False,
                          style="-o",
                          xlim=[truncation_dimensions[-1] / 1.2, truncation_dimensions[0] * 1.2])
        ax.set_xscale("log", basex=2)
        plt.title("Mean AUROC (over 963 side effects) by vector dimensionality")
        plt.xlabel("Dimensions")
        plt.ylabel("AUROC")

    return metrics


def apk(actual, predicted, k=50):
    """
    :param actual: true labels (0 or 1)
    :type actual: list
    :param predicted: predictions (between 0 and 1)
    :type predicted: list
    :param k:
    :type k: int
    :return: ap@k
    :rtype: float
    From decagon
    """
    from operator import itemgetter

    predicted = list(predicted)
    actual = list(actual)
    actual_new = []
    predicted_new = []
    edge_index = 0

    for i in range(len(predicted)):
        score = predicted[i]
        if actual[i] == 1:
            actual_new.append(edge_index)
        predicted_new.append((score, edge_index))
        edge_index += 1

    predicted_new = list(zip(*sorted(predicted_new, reverse=True, key=itemgetter(0))))[1]

    if not actual_new:
        return 0.0

    if len(predicted_new) > k:
        predicted_new = predicted_new[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted_new):
        if p in actual_new and p not in predicted_new[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual_new), k)


def incremental_analysis(file_location, positive_examples_file_name="positive_examples.tsv",
                         negative_examples_file_name="negative_examples.tsv", plot=True):
    positive_examples = pd.read_csv(file_location + positive_examples_file_name, sep="\t", header=None)
    negative_examples = pd.read_csv(file_location + negative_examples_file_name, sep="\t", header=None)

    positive_examples.columns = negative_examples.columns = ['subject_name', 'predicate_name', 'object_name']

    metrics = pd.DataFrame(
        columns=["auroc", "auprc", "mean_positive_score", "mean_negative_score"])
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        figure(num=None, figsize=(8, 6), dpi=120)

    i = 0
    while True:
        if not os.path.isfile(file_location + "negative_examples_scores_" + str(i) + ".txt"):
            break
        neg_scores = pd.read_csv(
            file_location + "negative_examples_scores_" + str(i) + ".txt", header=None)
        pos_scores = pd.read_csv(
            file_location + "positive_examples_scores_" + str(i) + ".txt", header=None)
        metrics = add_curve(
            positive_examples, negative_examples, pos_scores, neg_scores, i, metrics, plot=plot)
        i = i + 1
    if plot:
        plt.title("ROC by training cycle")

    metrics.index.name = "training_cycle"
    return metrics


def define_name_and_id_maps():
    """Load mappings of IDs used in decagon sets
    :return 1: dict mapping drug stitch IDs to drug names
    :return 2: dict mapping side effect stitch IDs to side effect names
    :return 3: dict mapping decagon drug IDs (0-644) to actual drug IDs (stitch IDs)
    :return 4: dict mapping decagon side effect IDs (0-1925) to actual side effect IDs (stitch IDs)
    :return 5: dict mapping decagon protein IDs (0-19080) to protein numbers used in original dataset.
    :rtype: (dict, dict, dict, dict, dict)
    """
    drug_names, side_effect_names = define_drug_and_side_effect_maps()

    decagon_drugs = pd.read_csv(
        "/home/haalbu/decagon/decagon_drugs.csv", header=None)
    decagon_drugs.columns = ["decagon_drug_id", "drug_id"]
    decagon_drugs_map = decagon_drugs.set_index(
        "decagon_drug_id").to_dict()['drug_id']
    decagon_proteins = pd.read_csv(
        "/home/haalbu/decagon/decagon_proteins.csv", header=None)
    decagon_proteins.columns = ["decagon_protein_id", "protein_id"]
    decagon_proteins_map = decagon_proteins.set_index("decagon_protein_id").to_dict()['protein_id']
    decagon_side_effects_regular = pd.read_csv(
        "/home/haalbu/decagon/decagon_side_effects.csv", header=None)
    decagon_side_effects_regular.columns = [
        "decagon_side_effect_id", "side_effect_id"]
    decagon_side_effects_regular["INV"] = False
    decagon_side_effects_inverses = decagon_side_effects_regular.copy()
    decagon_side_effects_inverses.INV = True

    number_of_side_effects = len(decagon_side_effects_regular)  # 963
    number_of_side_effects_including_inverses = number_of_side_effects * 2  # 1926
    decagon_side_effects_inverses.decagon_side_effect_id = [
        i for i in range(number_of_side_effects, number_of_side_effects_including_inverses)]

    # concatenate forward and backwards side effect relationships for one big list of relationships
    decagon_side_effects = decagon_side_effects_regular.append(
        decagon_side_effects_inverses).reset_index().drop(['index'], axis=1)
    decagon_side_effects['side_effect_id_combined'] = decagon_side_effects.apply(
        lambda row: row.side_effect_id if not row.INV else row.side_effect_id + "-INV", axis=1)
    decagon_side_effects_map = decagon_side_effects.set_index(
        "decagon_side_effect_id").to_dict()['side_effect_id_combined']

    return drug_names, side_effect_names, decagon_drugs_map, decagon_side_effects_map, decagon_proteins_map


def define_drug_and_side_effect_maps(transformed_names: bool = False):
    """Load drug and side effect names"""
    drug_names_df = pd.read_csv("/envme/decagon/drug_names.csv", header=None)
    drug_names_df.columns = ["drug_id", "drug_name"]
    drug_names = drug_names_df.set_index('drug_id').to_dict()['drug_name']

    if transformed_names:
        for drug_id in drug_names.keys():
            drug_name = drug_names[drug_id]
            drug_names[drug_id] = drug_name.replace(" ", "_").lower()

    side_effect_names_df = pd.read_csv(
        "/envme/decagon/side_effect_names.tsv", header=None, sep="\t")
    side_effect_names_df.columns = ["side_effect_id", "side_effect_name"]
    side_effect_names = side_effect_names_df.set_index(
        "side_effect_id").to_dict()['side_effect_name']

    if transformed_names:
        for side_effect_id in side_effect_names.keys():
            side_effect_name = side_effect_names[side_effect_id]
            side_effect_names[side_effect_id] = side_effect_name.replace(" ", "_").upper()

    return drug_names, side_effect_names


if __name__ == "__main__":
    file_location = "/envme/decagon/esp_decagon_deliverables/esp_16k_32cycles_new_negatives/"
    aurocs_by_se = get_aurocs_by_side_effect_files(file_location=file_location)
    print(aurocs_by_se)

