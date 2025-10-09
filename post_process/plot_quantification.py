import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


def plot_reg_quantification(csv_path, select_color,axes):
    df = pd.read_csv(csv_path)

    # Function to calculate ICC(2,1)
    def calculate_icc(data):
        n, k = data.shape
        mean_rater = data.mean(axis=0)
        mean_subject = data.mean(axis=1)
        grand_mean = data.values.flatten().mean()
        
        ss_total = ((data - grand_mean)**2).values.sum()
        ss_between = k * ((mean_subject - grand_mean)**2).sum()
        ss_rater = n * ((mean_rater - grand_mean)**2).sum()
        ss_error = ss_total - ss_between - ss_rater

        df_between = n - 1
        df_rater = k - 1
        df_error = df_between * df_rater

        ms_between = ss_between / df_between
        ms_rater = ss_rater / df_rater
        ms_error = ss_error / df_error

        icc = (ms_between - ms_error) / (ms_between + (k - 1)*ms_error + (k*(ms_rater - ms_error)/n))
        return icc

    # Get values
    x_vat = df['Label VAT (cm³)']
    y_vat = df['Pred VAT (cm³)']
    x_sat = df['Label SAT (cm³)']
    y_sat = df['Pred SAT (cm³)']

    # Linear regression
    slope_vat, intercept_vat, _, _, _ = linregress(x_vat, y_vat)
    slope_sat, intercept_sat, _, _, _ = linregress(x_sat, y_sat)

    # ICC
    icc_vat = calculate_icc(df[['Label VAT (cm³)', 'Pred VAT (cm³)']])
    icc_sat = calculate_icc(df[['Label SAT (cm³)', 'Pred SAT (cm³)']])



    # VAT plot
    axes[0].scatter(x_vat, y_vat,color=select_color) #, label='Data points')
    axes[0].plot(x_vat, slope_vat * x_vat + intercept_vat, color=select_color) #, label='Regression line')
    axes[0].set_title('VAT: Prediction vs Label')
    axes[0].set_xlabel('Referrence VAT (cm³)')
    axes[0].set_ylabel('Pred VAT (cm³)')
    print( f'ICC VAT = {icc_vat:.2f}')

    # SAT plot
    axes[1].scatter(x_sat, y_sat,color=select_color) #, label='Data points')
    axes[1].plot(x_sat, slope_sat * x_sat + intercept_sat, color=select_color) #, label='Regression line')
    axes[1].set_title('SAT: Prediction vs Label')
    axes[1].set_xlabel('Referrence SAT (cm³)')
    axes[1].set_ylabel('Pred SAT (cm³)')
    print( f'ICC SAT = {icc_sat:.2f}')


    plt.tight_layout()
    return [icc_vat,icc_sat]

select_color = ['#EA8379', '#299d8f','#E9C46A']
cmap = ListedColormap(select_color)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ICC = []
# ICC.append(plot_reg_quantification('../post_csv/FL_quantification.csv', select_color[0],axes))
# ICC.append(plot_reg_quantification('../post_csv/JT_quantification.csv', select_color[1], axes))
# ICC.append(plot_reg_quantification('../post_csv/P1_quantification.csv', select_color[2], axes))

ICC.append(plot_reg_quantification('../post_csv/Adult_FL_quantification.csv', select_color[1], axes))
ICC.append(plot_reg_quantification('../post_csv/Adult_Jt_quantification.csv', select_color[2], axes))
ICC.append(plot_reg_quantification('../post_csv/Adult_P1_quantification.csv', select_color[2], axes))

label_string = ['F1','J1','S1']

start_y = 0.95
line_spacing = 0.05  # Vertical space between lines
for i, value in enumerate(ICC):
    axes[0].text(
        0.05, 
        start_y - i * line_spacing, 
        f'{label_string[i]}: ICC = {value[0]:.2f}', 
        transform=axes[0].transAxes, 
        va='top'
    )

    axes[1].text(
        0.05, 
        start_y - i * line_spacing, 
        f'{label_string[i]}: ICC = {value[1]:.2f}', 
        transform=axes[1].transAxes, 
        va='top'
    )

legend_elements = [Patch(facecolor=cmap(i), label=label_string[i]) for i in [0,1,2]]
plt.legend(handles=legend_elements, loc='lower right')

plt.savefig('quantification.png')
plt.savefig('quantification.pdf')

