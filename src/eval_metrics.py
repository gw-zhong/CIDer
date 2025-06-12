import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_truth[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    a2 = accuracy_score(binary_truth, binary_preds)

    print(f'MAE: {mae:.3f}')
    print(f'Corr: {corr:.3f}')
    print(f'mult_acc_7: {mult_a7:.3f}')
    print(f'mult_acc_5: {mult_a5:.3f}')
    print(f'mult_acc_2: {a2:.3f}')
    print(f'F1_score: {f_score:.3f}')
    print('-' * 50)

    ans = {'MAE': mae, 'Corr': corr, 'mult_acc_7': mult_a7, 'mult_acc_5': mult_a5,
           'mult_acc_2': a2, 'F1_score': f_score}

    return ans


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_mosei_classification_binary(y_pred, y_true):
    """
    {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    # three classes
    y_pred_3 = np.argmax(y_pred, axis=1)
    Mult_acc_3 = accuracy_score(y_true, y_pred_3)
    F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
    # two classes
    y_pred_2 = np.array([[v[0], v[2]] for v in y_pred])
    # with 0 (< 0 or >= 0)
    y_pred_2 = np.argmax(y_pred_2, axis=1)
    y_true_2 = []
    for v in y_true:
        y_true_2.append(0 if v < 1 else 1)
    y_true_2 = np.array(y_true_2)
    Has0_acc_2 = accuracy_score(y_true_2, y_pred_2)
    Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
    # without 0 (< 0 or > 0)
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
    y_pred_2 = y_pred[non_zeros]
    y_pred_2 = np.argmax(y_pred_2, axis=1)
    y_true_2 = y_true[non_zeros]
    Non0_acc_2 = accuracy_score(y_true_2, y_pred_2)
    Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

    print(f'Has0_acc_2: {Has0_acc_2:.3f}')
    print(f'Has0_F1_score: {Has0_F1_score:.3f}')
    print(f'Non0_acc_2: {Non0_acc_2:.3f}')
    print(f'Non0_F1_score: {Non0_F1_score:.3f}')
    print(f'Acc_3: {Mult_acc_3:.3f}')
    print(f'F1_score_3: {F1_score_3:.3f}')
    print('-' * 50)

    ans = {
        "Has0_acc_2": Has0_acc_2,
        "Has0_F1_score": Has0_F1_score,
        "Non0_acc_2": Non0_acc_2,
        "Non0_F1_score": Non0_F1_score,
        "Acc_3": Mult_acc_3,
        "F1_score_3": F1_score_3
    }
    return ans


def eval_mosi_classification_binary(y_pred, y_true):
    return eval_mosei_classification_binary(y_pred, y_true)


def eval_mosei_classification_seven(y_pred, y_true):
    """
    {
        "Highly Negative": 0,
        "Negative": 1,
        "Weakly Negative": 2,
        "Neutral": 3,
        "Weakly Positive": 4,
        "Positive": 5,
        "Highly Positive": 6
    }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    # seven classes
    y_pred_7 = np.argmax(y_pred, axis=1)
    Mult_acc_7 = accuracy_score(y_true, y_pred_7)
    F1_score_7 = f1_score(y_true, y_pred_7, average='weighted')

    print(f'Acc_7: {Mult_acc_7:.3f}')
    print(f'F1_score_7: {F1_score_7:.3f}')

    ans = {
        "Acc_7": Mult_acc_7,
        "F1_score_7": F1_score_7
    }
    return ans


def eval_mosi_classification_seven(y_pred, y_true):
    return eval_mosei_classification_seven(y_pred, y_true)


def eval_mosei_classification_binary_cf(y_pred, y_cf_pred, y_true):
    """
    {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_cf_pred = y_cf_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    # three classes
    Mult_acc_3 = 0
    F1_score_3 = 0
    for tau in np.arange(0, 3.1, 0.1):
        tmp_y_pred = y_pred - tau * y_cf_pred
        tmp_y_pred_3 = np.argmax(tmp_y_pred, axis=1)
        tmp_Mult_acc_3 = accuracy_score(y_true, tmp_y_pred_3)
        tmp_F1_score_3 = f1_score(y_true, tmp_y_pred_3, average='weighted')
        Mult_acc_3 = max(Mult_acc_3, tmp_Mult_acc_3)
        F1_score_3 = max(F1_score_3, tmp_F1_score_3)
    # two classes
    Has0_acc_2 = 0
    Has0_F1_score = 0
    for tau in np.arange(0, 3.1, 0.1):
        tmp_y_pred = y_pred - tau * y_cf_pred
        tmp_y_pred_2 = np.array([[v[0], v[2]] for v in tmp_y_pred])
        # with 0 (< 0 or >= 0)
        tmp_y_pred_2 = np.argmax(tmp_y_pred_2, axis=1)
        y_true_2 = []
        for v in y_true:
            y_true_2.append(0 if v < 1 else 1)
        y_true_2 = np.array(y_true_2)
        tmp_Has0_acc_2 = accuracy_score(y_true_2, tmp_y_pred_2)
        tmp_Has0_F1_score = f1_score(y_true_2, tmp_y_pred_2, average='weighted')
        Has0_acc_2 = max(Has0_acc_2, tmp_Has0_acc_2)
        Has0_F1_score = max(Has0_F1_score, tmp_Has0_F1_score)
    # without 0 (< 0 or > 0)
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
    Non0_acc_2 = 0
    Non0_F1_score = 0
    for tau in np.arange(0, 3.1, 0.1):
        tmp_y_pred = y_pred - tau * y_cf_pred
        tmp_y_pred_2 = tmp_y_pred[non_zeros]
        tmp_y_pred_2 = np.argmax(tmp_y_pred_2, axis=1)
        y_true_2 = y_true[non_zeros]
        tmp_Non0_acc_2 = accuracy_score(y_true_2, tmp_y_pred_2)
        tmp_Non0_F1_score = f1_score(y_true_2, tmp_y_pred_2, average='weighted')
        Non0_acc_2 = max(Non0_acc_2, tmp_Non0_acc_2)
        Non0_F1_score = max(Non0_F1_score, tmp_Non0_F1_score)


    print(f'Non0_acc_2: {Non0_acc_2:.3f}')
    print(f'Non0_F1_score: {Non0_F1_score:.3f}')
    print('-' * 50)

    ans = {
        "Has0_acc_2": Has0_acc_2,
        "Has0_F1_score": Has0_F1_score,
        "Non0_acc_2": Non0_acc_2,
        "Non0_F1_score": Non0_F1_score,
        "Acc_3": Mult_acc_3,
        "F1_score_3": F1_score_3
    }
    return ans


def eval_mosi_classification_binary_cf(y_pred, y_cf_pred, y_true):
    return eval_mosei_classification_binary_cf(y_pred, y_cf_pred, y_true)


def eval_mosei_classification_seven_cf(y_pred, y_cf_pred, y_true):
    """
    {
        "Highly Negative": 0,
        "Negative": 1,
        "Weakly Negative": 2,
        "Neutral": 3,
        "Weakly Positive": 4,
        "Positive": 5,
        "Highly Positive": 6
    }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_cf_pred = y_cf_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    # seven classes
    Mult_acc_7 = 0
    F1_score_7 = 0
    for tau in np.arange(0, 3.1, 0.1):
        tmp_y_pred = y_pred - tau * y_cf_pred
        tmp_y_pred_7 = np.argmax(tmp_y_pred, axis=1)
        tmp_Mult_acc_7 = accuracy_score(y_true, tmp_y_pred_7)
        tmp_F1_score_7 = f1_score(y_true, tmp_y_pred_7, average='weighted')
        Mult_acc_7 = max(Mult_acc_7, tmp_Mult_acc_7)
        F1_score_7 = max(F1_score_7, tmp_F1_score_7)

    print(f'Acc_7: {Mult_acc_7:.3f}')
    print(f'F1_score_7: {F1_score_7:.3f}')
    print('-' * 50)

    ans = {
        "Acc_7": Mult_acc_7,
        "F1_score_7": F1_score_7
    }
    return ans


def eval_mosi_classification_seven_cf(y_pred, y_cf_pred, y_true):
    return eval_mosei_classification_seven_cf(y_pred, y_cf_pred, y_true)
