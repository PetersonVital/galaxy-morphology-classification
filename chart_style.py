import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


def apply_chart_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#DADADA",
            "axes.linewidth": 0.8,
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelcolor": "#2F2F2F",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": "#4A4A4A",
            "ytick.color": "#4A4A4A",
            "grid.color": "#CFCFCF",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.size": 10,
        }
    )


def create_figure(figsize=(10, 6)):
    apply_chart_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def format_axis(
    ax,
    title,
    subtitle="",
    xlabel="",
    ylabel="",
    rotate_xticks=0,
    y_as_percent=False,
    integer_y=False,
):
    ax.set_title(title, loc="left", pad=18)

    if subtitle:
        ax.text(
            0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#666666",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.25)
    ax.spines["bottom"].set_alpha(0.25)

    if rotate_xticks:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate_xticks)
            label.set_horizontalalignment("right")

    if y_as_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}" if x <= 1.05 else f"{x:.0f}%"))

    if integer_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def annotate_bars(ax, fmt="{:.0f}", suffix="", fontsize=9):
    for container in ax.containers:
        labels = []
        for bar in container:
            value = bar.get_height()
            labels.append(f"{fmt.format(value)}{suffix}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=fontsize, color="#333333")


def annotate_barh(ax, fmt="{:.0f}", suffix="", fontsize=9):
    for container in ax.containers:
        labels = []
        for bar in container:
            value = bar.get_width()
            labels.append(f"{fmt.format(value)}{suffix}")
        ax.bar_label(container, labels=labels, padding=4, fontsize=fontsize, color="#333333")


def save_figure(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
