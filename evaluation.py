"""
Evaluate model predictions and plot results. Code adapted / refactored from
Mitchell et al's detectGPT implementation.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

def get_roc_metrics(real_preds, sample_preds):
    """
    DESC: Calculates an evaluation metric known as area under the Receiver Operating Curve,
    i.e. AUROC. Intuitively, this metric measures the probability that given a random 
    human and a random chatGPT example, the model correctly guesses which is which.
    PARAMS: 
    real_preds, sample_preds: list of predictions (0 for human, 1 for chatGPT) over candidate passages
    RETURNS:
    a tuple:
        false_pos_rates: list of rates of false positives given different thresholds of number of pred. examples
        true_pos_rates: same as false_pos_rates, but with true positives
        AUROC: float with AUROC probability score
    """
    true_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    false_pos_rates, true_pos_rates, _ = roc_curve(true_labels, real_preds + sample_preds)
    roc_auc = auc(false_pos_rates, true_pos_rates)
    return false_pos_rates.tolist(), true_pos_rates.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_preds, sample_preds):
    """
    DESC: Calculate precision recall metric. Precision calculates number of accurate labels
    in positive classification group, i.e. true pos / all labeled pos. Recall is number of
    chatGPT ex's accurately labeled as such, i.e. true pos / (true pos + false neg).  
    PARAMS: 
    real_preds, sample_preds: list of predictions (0 for human, 1 for chatGPT) over candidate passages
    RETURNS:
    a tuple:
        precision: a list of precisions over diff probability thresholds
        recall: a list of recalls over diff probability thresholds
        PRAUC: area under precision recall curve 
    """
    true_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    precision, recall, _ = precision_recall_curve(true_labels, real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)



def save_roc_curves(experiments, query_model_name, save_dir):
    """
    DESC: Graph ROC curves for each experiment. Save them to save_dir.
    """
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({query_model_name} - T5-3B)')
    plt.legend(loc="lower right", fontsize=6)
    plt.show()
    if save_dir:
        plt.savefig(f"{save_dir}/roc_curves.png")

def save_ll_histograms(experiments, save_dir):
    """
    DESC: graph log likelihoods for perturbed vs. candidate examples, save to save_dir
    """
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            if save_dir:
                plt.savefig(f"{save_dir}/ll_histograms_{experiment['name']}.png")
        except:
            pass


def save_llr_histograms(experiments, save_dir):
    """
    DESC: same as save_ll_histograms, but also plot log likelihood RATIOS 
    """
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]
            
            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            if save_dir:
                plt.savefig(f"{save_dir}/llr_histograms_{experiment['name']}.png")
        except:
            pass

