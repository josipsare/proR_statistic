import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, shapiro, pearsonr, probplot
import seaborn as sns

dir_path = 'answers'

# Loop through each file in the directory
for filename in os.listdir(dir_path):
    file_path = os.path.join(dir_path, filename)

    # Load JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)

    bleu = []
    time = []
    answerLen = []
    for answer in data:
        bleu.append(answer['evaluation']['bleu_score'])
        answerLen.append(len(answer['answer']))
        time.append(answer['time'])
    
    # Calculate statistical parameters
    bleu_mean = np.mean(bleu)
    bleu_std = np.std(bleu)

    bleu_ks = kstest(bleu, 'norm')

    bleu_shapiro = shapiro(bleu)

    correlation_coefficient_len, correlation_p_value = pearsonr(bleu, answerLen)
    correlation_coefficient_time, correlation_p_value = pearsonr(bleu, time)

    # Save results to a file
    os.makedirs(filename[:-5], exist_ok=True)
    output_file_path = f"{filename[:-5]}/BLEUresults.txt" 
    with open(output_file_path, 'w') as output_file:
        output_file.write(f"{filename[:-5]}\n\n")
        output_file.write(f"-----------------------------------\n\n")

        output_file.write(f"Srednja vrijednost: {bleu_mean},\nStandardna devijacija duljine odgovora: {bleu_std}\n\n")

        output_file.write(f"-----------------------------------\n\n")

        output_file.write(f"Kolmogorov-Smirnovljev Test p-vrijednosti: {bleu_ks.pvalue}\n")

        output_file.write(f"Shapiro-Wilk Test p-vrijednosti: {bleu_shapiro.pvalue}\n")

        output_file.write(f"-----------------------------------\n\n")

        output_file.write(f"Pearsonov koeficjent koleracije za ocjenu i duljinu: {correlation_coefficient_len}\n")
        output_file.write(f"P-vrijednost koleracije: {correlation_p_value}\n\n")

        output_file.write(f"Pearsonov koeficjent koleracije za ocjenu i trajanje: {correlation_coefficient_time}\n")
        output_file.write(f"P-vrijednost koleracije: {correlation_p_value}\n\n")


    # Plotting the results
    labels = [f'q{i+1}' for i in range(len(bleu))]

    plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
    plt.bar(labels, bleu, color='skyblue')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Scores for Different Candidates')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Save the plot as a PNG image
    plt.tight_layout()  # Ensures labels fit within the figure
    plt.savefig(f"{filename[:-5]}/bleu_scores.png")


