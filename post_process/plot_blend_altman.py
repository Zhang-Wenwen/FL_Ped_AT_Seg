import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming your DataFrame is called df
# df = pd.read_csv("../post_csv/P1_PDFF.csv") 
df = pd.read_csv("../post_csv/Jt_PDFF.csv") 
# df = pd.read_csv("../post_csv/FL_PDFF.csv")

def bland_altman_plot(data1, data2, title, save_path):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, axis=0)

    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Create plot
    plt.figure(figsize=(7, 5))
    plt.scatter(mean, diff, alpha=0.6)
    plt.axhline(mean_diff, color='blue', linestyle='--', label=f'Mean Diff: {mean_diff:.4f}')
    plt.axhline(loa_upper, color='red', linestyle='--', label=f'+1.96 SD: {loa_upper:.4f}')
    plt.axhline(loa_lower, color='red', linestyle='--', label=f'-1.96 SD: {loa_lower:.4f}')
    plt.grid(True)
    plt.savefig(f'{save_path}.pdf')

    plt.title(title)
    plt.xlabel('Mean of Reference and Prediction')
    plt.ylabel('Difference (Reference - Prediction)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}.png')

# Plot for VAT
save_path = 'PDFF_VAT_jt_90'
bland_altman_plot(
    df['PDFF VAT Referrence'],
    df['PDFF VAT Predictions'],
    'Bland-Altman Plot: VAT Ref vs VAT Pred',
    save_path,
)

# Plot for SAT
save_path = 'PDFF_SAT_jt_90'
bland_altman_plot(
    df['PDFF SAT Referrence'],
    df['PDFF SAT Predictions'],
    'Bland-Altman Plot: SAT Ref vs SAT Pred',
    save_path,
)
