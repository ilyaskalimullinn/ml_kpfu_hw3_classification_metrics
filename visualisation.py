import pandas as pd
from plotly import express as px


def plot_precision_recall_curve(metrics: pd.DataFrame, round_digits: int = 2, plot_title: str = 'Precision-Recall curve', save_path: str = None):

    round_str = f":.{round_digits}r"

    fig = px.line(
        metrics,
        x='recall',
        y='precision',
        hover_data={'accuracy': round_str,
                    'threshold': round_str,
                    'f1_score': round_str,
                    'precision': round_str,
                    'recall': round_str}
    )

    fig.update_layout(
        title=plot_title,
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    fig.show()
    if save_path:
        fig.write_html(save_path)


def plot_roc_curve(metrics: pd.DataFrame, round_digits: int = 2, plot_title: str = 'Precision-Recall curve', save_path: str = None):
    round_str = f":.{round_digits}r"

    fig = px.line(
        metrics,
        x='FPR',
        y='recall',
        hover_data={'accuracy': round_str,
                    'threshold': round_str,
                    'f1_score': round_str,
                    'FPR': round_str,
                    'recall': round_str}
    )

    fig.update_layout(
        title=plot_title,
        xaxis_title="FPR",
        yaxis_title="Recall"
    )

    fig.show()
    if save_path:
        fig.write_html(save_path)
