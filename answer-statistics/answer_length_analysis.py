import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os


files = ['answers/llama-2-7b-chat.Q5_K_M.gguf.json', 'answers/mistral-7b-instruct-v0.1.Q5_K_M.gguf.json', 'answers/mistral-7b-openorca.Q4_0.gguf.json', 'answers/zephyr-7b-alpha.Q5_K_M.gguf.json', 'answers/zephyr-7b-beta.Q5_K_M.gguf.json']

def process_file(file_name):
    print(f"Processing file: {file_name}")
    with open(file_name, 'r') as file:
        data = json.load(file)
    print("Data loaded successfully.")

    records = []
    for item in data:
        answer_length = len(item['answer'].split())
        bleu_score = item['evaluation']['bleu_score']
        rouge_score = item['evaluation']['rouge_score']['f1']

        overall_quality_scores = [float(value) for value in item['evaluation']['human_eval']['overall_quality'].values() if value != '']
        if overall_quality_scores:
            avg_human_eval = sum(overall_quality_scores) / len(overall_quality_scores)
        else:
            avg_human_eval = 0  

        records.append({'AnswerLength': answer_length, 'BLEUScore': bleu_score, 'ROUGEScore': rouge_score, 'HumanEvaluation': avg_human_eval})


    df = pd.DataFrame(records)

    correlation_matrix = df.corr(method='spearman')
    print(f"Correlation Matrix for {file_name}:\n", correlation_matrix)

    pairplot = sns.pairplot(df)
    pairplot.fig.suptitle(f'Pairplot of Metrics for {file_name}', y=1.02)
    pairplot.savefig(f"{file_name}_pairplot.png")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    sns.scatterplot(data=df, x='AnswerLength', y='BLEUScore', ax=axs[0]).set_title('BLEU Score vs Answer Length')
    sns.scatterplot(data=df, x='AnswerLength', y='ROUGEScore', ax=axs[1]).set_title('ROUGE Score vs Answer Length')
    sns.scatterplot(data=df, x='AnswerLength', y='HumanEvaluation', ax=axs[2]).set_title('Human Evaluation vs Answer Length')
    plt.tight_layout()
    fig.savefig(f"{file_name}_scatterplots.png")

    print(f"Finished processing {file_name}")


for file_name in files:
    if os.path.exists(file_name):
        process_file(file_name)
    else:
        print(f"File {file_name} not found.")

print("All files processed.")
