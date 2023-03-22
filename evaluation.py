"""
Evaluate model predictions and plot results. Code adapted from
Mitchell et al's detectGPT implementation.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc


MODELS = ['gpt-j-6B', 'gpt-neo-2.7B', 'gpt2', 'opt-2.7b']

TEMPERATURES = ['00', '50', '100']

COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def plot_against_temperature(dataset):
    with plt.ion():
        plt.figure()
        x = ['0.0', '0.5', '1.0']
        for model, color in zip(MODELS, COLORS):
            dfs = [pd.read_csv(f'Results/{dataset}_t{temp}_n100/{model}/scores.csv', index_col=0) for temp in TEMPERATURES]
            y = [df['AUROC']['z'] for df in dfs]
            plt.plot(x, y, color=color, label=model)
        plt.legend(loc='upper right')
        plt.xlabel('Temperature')
        plt.ylabel('AUROC')
        dataname = 'XSum' if dataset == 'xsum' else 'WritingPrompts'
        plt.title(f'Temperature vs. AUROC over {dataname} Dataset')
        plt.savefig(f'{dataset}_vs_temp.pdf')
        plt.clf()



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



def save_roc_curves(experiments, save_dir):
    """
    DESC: Graph ROC curves for each experiment. Save them to save_dir.
    """
    # first, clear plt
    plt.figure()

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
    plt.title(f'ROC Curves')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{save_dir}/roc_curves.png")
    plt.show()


def save_ll_histograms(results, save_dir):
    """
    DESC: graph log likelihoods for perturbed vs. candidate examples, save to save_dir
    """
    # first, clear plt
    plt.clf()

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
    plt.savefig(f"{save_dir}/ll_histograms.pdf")
        


def save_llr_histograms(results, save_dir):
    """
    DESC: same as save_ll_histograms, but also plot log likelihood RATIOS 
    """
    # first, clear plt
    plt.clf()

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
    plt.savefig(f"{save_dir}/llr_histograms.pdf")


def save_scores(experiments, save_dir):
    """
    Save AUROC and PRAUC scores for discrepancy and z-score criteria.
    Assumes that the z-score experiment is the first one in experiments. 
    """
    scores = pd.DataFrame(data={'AUROC': [experiment['metrics']['roc_auc'] for experiment in experiments],
                                'PRAUC': [experiment['pr_metrics']['pr_auc'] for experiment in experiments]},
                          index=['z','d'])
    scores.to_csv(f'{save_dir}/scores.csv')
    return scores

