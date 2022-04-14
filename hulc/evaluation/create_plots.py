import argparse
import json
import os.path
from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import seaborn as sns

# plt.rcParams['text.usetex'] = True
from hulc.utils.utils import format_sftp_path

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"
plt.pyplot.title(r"ABC123 vs $\mathrm{ABC123}^{123}$")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

task_categories = {
    "rotate_red_block_right": "dynamic",
    "rotate_red_block_left": "dynamic",
    "rotate_blue_block_right": "dynamic",
    "rotate_blue_block_left": "dynamic",
    "rotate_pink_block_right": "dynamic",
    "rotate_pink_block_left": "dynamic",
    "push_red_block_right": "dynamic",
    "push_red_block_left": "dynamic",
    "push_blue_block_right": "dynamic",
    "push_blue_block_left": "dynamic",
    "push_pink_block_right": "dynamic",
    "push_pink_block_left": "dynamic",
    "move_slider_left": "static",
    "move_slider_right": "static",
    "open_drawer": "static",
    "close_drawer": "static",
    "lift_red_block_table": "dynamic",
    "lift_blue_block_table": "dynamic",
    "lift_pink_block_table": "dynamic",
    "lift_red_block_slider": "dynamic",
    "lift_blue_block_slider": "dynamic",
    "lift_pink_block_slider": "dynamic",
    "lift_red_block_drawer": "dynamic",
    "lift_blue_block_drawer": "dynamic",
    "lift_pink_block_drawer": "dynamic",
    "place_in_slider": "static",
    "place_in_drawer": "static",
    "turn_on_lightbulb": "static",
    "turn_off_lightbulb": "static",
    "turn_on_led": "static",
    "turn_off_led": "static",
    "push_into_drawer": "dynamic",
    "stack_block": "dynamic",
    "unstack_block": "dynamic",
}

task_classes = {
    "rotate_red_block_right": 1,
    "rotate_red_block_left": 1,
    "rotate_blue_block_right": 1,
    "rotate_blue_block_left": 1,
    "rotate_pink_block_right": 1,
    "rotate_pink_block_left": 1,
    "push_red_block_right": 2,
    "push_red_block_left": 2,
    "push_blue_block_right": 2,
    "push_blue_block_left": 2,
    "push_pink_block_right": 2,
    "push_pink_block_left": 2,
    "move_slider_left": 3,
    "move_slider_right": 3,
    "open_drawer": 4,
    "close_drawer": 4,
    "lift_red_block_table": 5,
    "lift_blue_block_table": 5,
    "lift_pink_block_table": 5,
    "lift_red_block_slider": 5,
    "lift_blue_block_slider": 5,
    "lift_pink_block_slider": 5,
    "lift_red_block_drawer": 5,
    "lift_blue_block_drawer": 5,
    "lift_pink_block_drawer": 5,
    "place_in_slider": 6,
    "place_in_drawer": 6,
    "turn_on_lightbulb": 7,
    "turn_off_lightbulb": 7,
    "turn_on_led": 8,
    "turn_off_led": 8,
    "push_into_drawer": 6,
    "stack_block": 9,
    "unstack_block": 9,
}

markers_plans = {
    "rotate_red_block_right": "o",
    "rotate_red_block_left": "P",
    "rotate_blue_block_right": "s",
    "rotate_blue_block_left": "X",
    "rotate_pink_block_right": "D",
    "rotate_pink_block_left": "^",
    "push_red_block_right": "o",
    "push_red_block_left": "P",
    "push_blue_block_right": "s",
    "push_blue_block_left": "X",
    "push_pink_block_right": "D",
    "push_pink_block_left": "^",
    "move_slider_left": "o",
    "move_slider_right": "P",
    "open_drawer": "s",
    "close_drawer": "X",
    "lift_red_block_table": "o",
    "lift_red_block_slider": "P",
    "lift_red_block_drawer": "s",
    "lift_blue_block_table": "o",
    "lift_blue_block_slider": "P",
    "lift_blue_block_drawer": "s",
    "lift_pink_block_table": "o",
    "lift_pink_block_slider": "P",
    "lift_pink_block_drawer": "s",
    "place_in_slider": "P",
    "place_in_drawer": "s",
    "turn_on_lightbulb": "X",
    "turn_off_lightbulb": "s",
    "turn_on_led": "^",
    "turn_off_led": "P",
    "push_into_drawer": "o",
    "stack_block": "X",
    "unstack_block": "D",
}


def load_eval_data(path):
    with open(path) as f:
        return json.load(f)


def load_results(training_dirs):
    results = {}
    for training_dir in training_dirs:
        training_dir = format_sftp_path(training_dir)
        name = training_dir.name
        eval_file = training_dir / "evaluation/results.json"
        if not eval_file.exists():
            continue
        results[name] = load_eval_data(eval_file)
    return results


def load_tsne_data(training_dirs):
    tsne_data = {}
    for training_dir in training_dirs:
        training_dir = format_sftp_path(training_dir)
        name = training_dir.stem
        if len(list((training_dir / "evaluation").glob("tsne_data*"))):
            tsne_data[name] = {}
        for path in (training_dir / "evaluation").glob("tsne_data*"):
            epoch = int(path.stem.split("_")[2])
            try:
                tsne_data[name][epoch] = np.load(training_dir / f"evaluation/tsne_data_{epoch}.npz")
            except FileNotFoundError:
                continue
    return tsne_data


def plot_avg_seq_len(results, labels):
    epochs = [[int(x) for x in sorted(result, key=int)] for result in results.values()]
    avg_seq_lens = [[result[x]["avg_seq_len"] for x in sorted(result, key=int)] for result in results.values()]
    ranking = np.argsort([max(x) for x in avg_seq_lens])[::-1]
    epochs = np.array(epochs, dtype=object)[ranking]
    avg_seq_lens = np.array(avg_seq_lens, dtype=object)[ranking]
    labels = np.array(labels)[ranking]
    plot_curves(epochs, avg_seq_lens, labels, "Epochs", "Avg. sequence length", save_path="/tmp/avg_seq_len.pdf")


def plot_chain5(results, labels):
    epochs = [[int(x) for x in sorted(result, key=int)] for result in results.values()]
    chain5_sr = [[result[x]["chain_sr"]["5"] * 100 for x in sorted(result, key=int)] for result in results.values()]
    # ?chain5_sr = np.array(chain5_sr)
    # ranking = np.argsort(np.max(chain5_sr, axis=1))[::-1]
    ranking = np.argsort([max(x) for x in chain5_sr])[::-1]
    epochs = np.array(epochs, dtype=object)[ranking]
    chain5_sr = np.array(chain5_sr, dtype=object)[ranking]
    labels = np.array(labels)[ranking]
    plot_curves(epochs, chain5_sr, labels, "Epochs", "Chain 5 SR %", save_path="/tmp/chain5.pdf")


def plot_chain_sr(results, labels):
    x_labels = []
    chain_success_rates = []
    for result in results.values():
        best_model = max([(epoch, v["avg_seq_len"]) for epoch, v in result.items()], key=lambda x: x[1])[0]
        chain_success_rates.append(np.array(list(result[best_model]["chain_sr"].values())) * 100)
        x_labels.append(result[best_model]["chain_sr"].keys())
    ranking = np.argsort(np.array(chain_success_rates)[:, -1])[::-1]
    x_labels = np.array(x_labels)[ranking]
    chain_success_rates = np.array(chain_success_rates)[ranking]
    labels = np.array(labels)[ranking]
    plot_curves(
        x_labels,
        chain_success_rates,
        labels,
        "Number of instructions in a row",
        "Tasks Completed %",
        save_path="/tmp/chain_sr.pdf",
        marker="o",
        linewidth=2,
    )


def plot_curves(
    x_values_lists,
    y_values_lists,
    run_labels,
    x_label,
    y_label,
    save_path,
    marker=None,
    linewidth=2,
):
    fig = plt.figure(figsize=(16, 10))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]  # , "#17becf"]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    ax = plt.subplot()  # Defines ax variable by creating an empty plot
    # Set the tick labels font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        # label.set_fontname('Verdana')
        label.set_fontsize(28)
    for i, (x_values, y_values, label) in enumerate(zip(x_values_lists, y_values_lists, run_labels)):
        plt.plot(
            x_values,
            y_values,
            label=label,
            color=colors[i % len(colors)],
            linewidth=linewidth,
            marker=marker,
            markersize=10,
            ls=linestyles[i % len(linestyles)],
        )
    # axis_font = {'fontname': 'Verdana', 'size': '32'}
    axis_font = {"size": "32"}
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0, prop={"size": 18})
    plt.xlabel(x_label, **axis_font)
    plt.ylabel(y_label, **axis_font)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_task_sr(results, labels):
    num_trainings = len(results)
    x_labels = []
    task_success_rates = []
    scores = []
    tasks_total = []
    for result in results.values():
        best_epoch, best_score = max([(epoch, v["avg_seq_len"]) for epoch, v in result.items()], key=lambda x: x[1])
        scores.append(best_score)
        task_success_rates.append(
            {k: v["success"] / v["total"] * 100 for k, v in result[best_epoch]["task_info"].items()}
        )
        tasks_total.append({k: v["total"] for k, v in result[best_epoch]["task_info"].items()})

    ranking = np.argsort(scores)[::-1]
    tasks = [task for task, total in tasks_total[ranking[0]].items() if total >= 10]
    tasks_sorted = sorted(tasks, key=lambda x: task_success_rates[ranking[0]].get(x, 0), reverse=True)

    n_subplots = 4
    fig, axs = plt.subplots(n_subplots, figsize=(32, 32))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]

    width = 1 / (num_trainings + 1)
    for j, ax in enumerate(axs):
        tasks_sorted_split = np.array_split(tasks_sorted, n_subplots)[j]
        for i, rank in enumerate(ranking):
            x = np.arange((len(tasks_sorted_split))) + i * width
            y = [task_success_rates[rank].get(task, 0) for task in tasks_sorted_split]
            ax.bar(x, y, label=labels[rank], color=colors[i % len(colors)], width=(1 / (num_trainings + 1)))
        for label in ax.get_yticklabels():
            # label.set_fontname('Verdana')
            label.set_fontsize(28)
        ax.set_xticks(np.arange(len(tasks_sorted_split)) + width * num_trainings / 2)
        ax.set_xticklabels(
            labels=[task.replace("_", " ").capitalize() for task in tasks_sorted_split],
            fontsize=20,
            ha="center",
            rotation=0,
        )
        ax.xaxis.set_tick_params(length=0)
        ax.set_ylim([0, 100])
        axis_font = {"size": "32"}
        ax.set_ylabel("Success Rate %", **axis_font)
    plt.legend(bbox_to_anchor=(1.01, 4.6), loc=2, borderaxespad=0.0, prop={"size": 18})
    fig.savefig("/tmp/task_sr.pdf", dpi=300, bbox_inches="tight")


def plot_task_categories(results, labels):
    static_task_success_rates = []
    dynamic_task_success_rates = []
    scores = []
    for result in results.values():
        best_epoch, best_score = max([(epoch, v["avg_seq_len"]) for epoch, v in result.items()], key=lambda x: x[1])
        scores.append(best_score)
        success_static = np.array(
            [
                sum([v["success"] for k, v in result[i]["task_info"].items() if task_categories[k] == "static"])
                for i in sorted(result.keys(), key=int)
            ]
        )
        total_static = np.array(
            [
                sum([v["total"] for k, v in result[i]["task_info"].items() if task_categories[k] == "static"])
                for i in sorted(result.keys(), key=int)
            ]
        )
        static_task_success_rates.append(success_static / total_static * 100)

        success_dynamic = np.array(
            [
                sum([v["success"] for k, v in result[i]["task_info"].items() if task_categories[k] == "dynamic"])
                for i in sorted(result.keys(), key=int)
            ]
        )
        total_dynamic = np.array(
            [
                sum([v["total"] for k, v in result[i]["task_info"].items() if task_categories[k] == "dynamic"])
                for i in sorted(result.keys(), key=int)
            ]
        )
        dynamic_task_success_rates.append(success_dynamic / total_dynamic * 100)

    ranking = np.argsort(scores)[::-1]
    labels = [labels[i] for i in ranking]
    dynamic_task_success_rates = [dynamic_task_success_rates[i] for i in ranking]
    static_task_success_rates = [static_task_success_rates[i] for i in ranking]
    fig = plt.figure(figsize=(16, 10))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]  # , "#17becf"]
    ax = plt.subplot()  # Defines ax variable by creating an empty plot
    # Set the tick labels font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        # label.set_fontname('Verdana')
        label.set_fontsize(28)
    for i, (dynamic, static, label) in enumerate(zip(dynamic_task_success_rates, static_task_success_rates, labels)):
        plt.plot(
            np.arange(len(static)),
            static,
            label=f"{label} static",
            color=colors[i % len(colors)],
            linewidth=2,
            ls="solid",
        )
        plt.plot(
            np.arange(len(dynamic)),
            dynamic,
            label=f"{label} dynamic",
            color=colors[i % len(colors)],
            linewidth=2,
            ls="dashed",
        )
    # axis_font = {'fontname': 'Verdana', 'size': '32'}
    axis_font = {"size": "32"}
    handles = [mlines.Line2D([], [], color=colors[i % len(colors)], label=labels[i]) for i in range(len(labels))]
    handles += [
        mlines.Line2D([], [], color="black", label=label, linestyle=style)
        for label, style in zip(("static", "dynamic"), ("solid", "dashed"))
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0, prop={"size": 18})
    plt.xlabel("Epochs", **axis_font)
    plt.ylabel("Success rate %", **axis_font)
    fig.savefig("/tmp/task_categories.pdf", dpi=300, bbox_inches="tight")


def plot_ranking(results, labels):
    num_trainings = len(results)
    avg_seq_lens = []
    for result in results.values():
        best_model = max([v["avg_seq_len"] for v in result.values()])
        avg_seq_lens.append(best_model)

    ranking = np.argsort(avg_seq_lens)[::-1]
    avg_seq_lens = np.array(avg_seq_lens)[ranking]
    labels = np.array(labels)[ranking]
    fig = plt.figure(figsize=(32, 8))
    ax = plt.subplot()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]

    x = np.arange(num_trainings) / 4
    y = avg_seq_lens
    ax.bar(x, y, color=colors[0], width=0.2)
    for label in ax.get_yticklabels():
        # label.set_fontname('Verdana')
        label.set_fontsize(28)
    ax.set_xticks(x)
    ax.set_xticklabels(labels=[name.replace("_", " ").capitalize() for name in labels], fontsize=20, ha="right")
    ax.xaxis.set_tick_params(length=0)
    plt.xticks(rotation=70)

    axis_font = {"size": "32"}
    plt.legend(loc="upper right", prop={"size": 18})
    plt.ylabel("Average Sequence Length", **axis_font)

    fig.savefig("/tmp/best_performance.pdf", dpi=300, bbox_inches="tight")


def create_tsne_plot(results, tsne_data_dict):
    for name, tsne_data_training in tsne_data_dict.items():
        result = results[name]
        epoch, _ = max([(int(epoch), v["avg_seq_len"]) for epoch, v in result.items()], key=lambda x: x[1])
        tsne_data = tsne_data_training[epoch]
        ids, labels, latent_goals, plans = (tsne_data[x] for x in ["ids", "labels", "latent_goals", "plans"])
        order = list(
            list(
                zip(
                    *sorted(
                        zip(range(len(labels)), labels),
                        key=lambda x: (task_classes.get(x[1]), list(task_classes.keys()).index(x[1])),
                    )
                )
            )[0]
        )
        # ids = ids[order]
        labels = labels[order]
        # latent_goals = latent_goals[order]
        plans = plans[order]

        x_tsne = TSNE(perplexity=10, n_jobs=8).fit_transform(plans)

        fig = plt.figure(figsize=(10, 10))
        plt.rcParams["font.size"] = "16"
        unique_categories = list(set(task_classes.values()))
        palette = sns.color_palette("bright", len(unique_categories))
        color_dict = {key: palette[unique_categories.index(value)] for key, value in task_classes.items()}
        g = sns.scatterplot(
            x=x_tsne[:, 0].flatten(),
            y=x_tsne[:, 1].flatten(),
            hue=labels,
            palette=color_dict,
            legend="full",
            alpha=1,
            style=labels,
            markers=markers_plans,
        )
        sns.despine(left=True, bottom=True, right=True)
        g.set(yticks=[])
        g.set(xticks=[])
        plt.legend(bbox_to_anchor=(1.01, 1), ncol=2, loc=2, borderaxespad=0.0)
        fig.savefig(f"/tmp/latent_plans_{name}_{epoch}.pdf", dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot the training results for multiple models")
    parser.add_argument("--training_dirs", type=str, default=None, help="Comma separated training dirs")
    parser.add_argument("--parent_dirs", type=str, default=None, help="Comma separated parent dirs of training dirs.")
    parser.add_argument("--labels", type=str, default=None, help="Comma separated parent dirs of labels.")
    args = parser.parse_args()

    if args.training_dirs is not None:
        training_dirs = [Path(path) for path in args.training_dirs.split(",")]
    elif args.parent_dirs is not None:
        training_dirs = [
            Path(path)
            for parent_dir in args.parent_dirs.split(",")
            for path in format_sftp_path(Path(parent_dir)).glob("*")
            if os.path.isdir(path)
        ]
    else:
        print("Please set either --training_dirs or --parent_dirs")
        raise Exception

    results = load_results(training_dirs)

    if args.labels is not None:
        labels = [label for label in args.labels.split(",")]
        if len(labels) != len(results.keys()):
            print("Wrong number of labels!")
            labels = [label.split("_", maxsplit=1)[1].replace("_", " ") for label in results.keys()]
    else:
        labels = [label.split("_", maxsplit=1)[1].replace("_", " ") for label in results.keys()]
    #
    plot_chain_sr(results, labels)
    plot_chain5(results, labels)
    plot_avg_seq_len(results, labels)
    plot_task_sr(results, labels)
    plot_task_categories(results, labels)
    plot_ranking(results, labels)

    # tsne_data_dict = load_tsne_data(training_dirs)
    # create_tsne_plot(results, tsne_data_dict)


if __name__ == "__main__":
    main()
