import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def calculate_ks(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    data = data.sort_values(by='y_pred_proba', ascending=False)
    
    data['cum_event_rate'] = np.cumsum(data['y_true']) / sum(data['y_true'])
    data['cum_non_event_rate'] = np.cumsum(1 - data['y_true']) / sum(1 - data['y_true'])
    
    ks_statistic = max(abs(data['cum_event_rate'] - data['cum_non_event_rate']))
    
    return ks_statistic


def calculate_woe_iv(df, feature, target):
    df = df[[feature, target]].copy()
    df['non_target'] = 1 - df[target]
    
    grouped = df.groupby(feature).agg({target: 'sum', 'non_target': 'sum'})
    grouped['total'] = grouped[target] + grouped['non_target']
    grouped['percentage'] = grouped['total'] / grouped['total'].sum()
    grouped['non_target_rate'] = grouped['non_target'] / grouped['non_target'].sum()
    grouped['target_rate'] = grouped[target] / grouped[target].sum()
    grouped['WOE'] = np.log(grouped['non_target_rate'] / grouped['target_rate'])
    grouped['IV'] = (grouped['non_target_rate'] - grouped['target_rate']) * grouped['WOE']
    
    return grouped['WOE'], grouped['IV'].sum()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def calculate_fairness_metrics(y_true, y_pred, sensitive_attribute):
    subgroup_1 = sensitive_attribute == 0  
    subgroup_2 = sensitive_attribute == 1  

    dp_1 = np.mean(y_pred[subgroup_1])
    dp_2 = np.mean(y_pred[subgroup_2])
    demographic_parity_diff = abs(dp_1 - dp_2)

    tpr_1 = np.sum((y_true[subgroup_1] == 1) & (y_pred[subgroup_1] == 1)) / np.sum(y_true[subgroup_1] == 1)
    tpr_2 = np.sum((y_true[subgroup_2] == 1) & (y_pred[subgroup_2] == 1)) / np.sum(y_true[subgroup_2] == 1)
    equal_opportunity_diff = abs(tpr_1 - tpr_2)

    fpr_1 = np.sum((y_true[subgroup_1] == 0) & (y_pred[subgroup_1] == 1)) / np.sum(y_true[subgroup_1] == 0)
    fpr_2 = np.sum((y_true[subgroup_2] == 0) & (y_pred[subgroup_2] == 1)) / np.sum(y_true[subgroup_2] == 0)
    equalized_odds_diff = abs(tpr_1 - tpr_2) + abs(fpr_1 - fpr_2)

    disparate_impact_ratio = dp_1 / dp_2 if dp_2 != 0 else np.inf

    return {
        'demographic_parity_diff': demographic_parity_diff,
        'equal_opportunity_diff': equal_opportunity_diff,
        'equalized_odds_diff': equalized_odds_diff,
        'disparate_impact_ratio': disparate_impact_ratio
    }