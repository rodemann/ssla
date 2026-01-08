from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct

from src.utils import CI, coverage


def coverage_to_tex_table(
    dm,
    ssla_kfac,
    ssla_diag,
    ssla_dense,
    assla_kfac,
    assla_diag,
    assla_dense,
    la_kfac,
    la_diag,
    la_dense,
    vi_pred,
    mfvi_pred,
    hmc_pred,
    target: Path,
):
    tab = rf"""\begin{{table*}}[ht]\label{{tab:hetero_coverage}}
    \centering
    \resizebox{{1\textwidth}}{{!}}{{
        \begin{{tabular}}{{l|ccc|ccc|ccc|cc|c}}
        & \multicolumn{{3}}{{c}}{{SSLA}} & \multicolumn{{3}}{{|c|}}{{ASSLA}} & \multicolumn{{3}}{{c}}{{LA}} & \multicolumn{{2}}{{|c|}}{{VI}} & \multicolumn{{1}}{{|c}}{{MCMC}} \\ \midrule
        CI & KFAC & DIAG & DENSE & KFAC & DIAG & DENSE & KFAC & DIAG & DENSE & VI & MFVI & HMC \\ \midrule
        \hline 
        95\% & {coverage(ssla_kfac, dm, CI.NINTY_FIVE) if ssla_kfac else -99:.1f} & {coverage(ssla_diag, dm,  CI.NINTY_FIVE) if ssla_diag else -99:.1f} & {coverage(ssla_dense, dm,  CI.NINTY_FIVE) if ssla_dense else -99:.1f} & {coverage(assla_kfac, dm, CI.NINTY_FIVE) if assla_kfac else -99:.1f} & {coverage(assla_diag, dm, CI.NINTY_FIVE) if assla_diag else -99:.1f} & {coverage(assla_dense, dm, CI.NINTY_FIVE) if assla_dense else -99:.1f} & {coverage(la_kfac, dm,  CI.NINTY_FIVE):.1f} & {coverage(la_diag, dm,  CI.NINTY_FIVE):.1f} & {coverage(la_dense, dm,  CI.NINTY_FIVE):.1f} & {coverage(vi_pred, dm, CI.NINTY_FIVE):.1f} & {coverage(mfvi_pred, dm, CI.NINTY_FIVE):.1f} & {coverage(hmc_pred, dm, CI.NINTY_FIVE):.1f} \\
        90\% & {coverage(ssla_kfac, dm,  CI.NINTY) if ssla_kfac else -99:.1f} & {coverage(ssla_diag, dm,  CI.NINTY) if ssla_diag else -99:.1f} & {coverage(ssla_dense, dm,  CI.NINTY) if ssla_dense else -99:.1f} & {coverage(assla_kfac, dm, CI.NINTY) if assla_kfac else -99:.1f} & {coverage(assla_diag, dm, CI.NINTY) if assla_diag else -99:.1f} & {coverage(assla_dense, dm, CI.NINTY) if assla_dense else -99:.1f} & {coverage(la_kfac, dm,  CI.NINTY):.1f} & {coverage(la_diag, dm,  CI.NINTY):.1f} & {coverage(la_dense, dm,  CI.NINTY):.1f} & {coverage(vi_pred, dm, CI.NINTY):.1f} & {coverage(mfvi_pred, dm, CI.NINTY):.1f} & {coverage(hmc_pred, dm, CI.NINTY):.1f} \\
        75\% & {coverage(ssla_kfac, dm,  CI.SEVENTY_FIVE) if ssla_kfac else -99:.1f} & {coverage(ssla_diag, dm,  CI.SEVENTY_FIVE) if ssla_diag else -99:.1f} & {coverage(ssla_dense, dm,  CI.SEVENTY_FIVE) if ssla_dense else -99:.1f} & {coverage(assla_kfac, dm, CI.SEVENTY_FIVE) if assla_kfac else -99:.1f} & {coverage(assla_diag, dm, CI.SEVENTY_FIVE) if assla_diag else -99:.1f} & {coverage(assla_dense, dm, CI.SEVENTY_FIVE) if assla_dense else -99:.1f} & {coverage(la_kfac, dm,  CI.SEVENTY_FIVE):.1f} & {coverage(la_diag, dm,  CI.SEVENTY_FIVE):.1f} & {coverage(la_dense, dm,  CI.SEVENTY_FIVE):.1f} & {coverage(vi_pred, dm, CI.SEVENTY_FIVE):.1f} & {coverage(mfvi_pred, dm, CI.SEVENTY_FIVE):.1f} & {coverage(hmc_pred, dm, CI.SEVENTY_FIVE):.1f} \\
        50\% & {coverage(ssla_kfac, dm,  CI.FIFTY) if ssla_kfac else -99:.1f} & {coverage(ssla_diag, dm,  CI.FIFTY) if ssla_diag else -99:.1f} & {coverage(ssla_dense, dm,  CI.FIFTY) if ssla_dense else -99:.1f} & {coverage(assla_kfac, dm, CI.FIFTY) if assla_kfac else -99:.1f} & {coverage(assla_diag, dm, CI.FIFTY) if assla_diag else -99:.1f} & {coverage(assla_dense, dm, CI.FIFTY) if assla_dense else -99:.1f} & {coverage(la_kfac, dm,  CI.FIFTY):.1f} & {coverage(la_diag, dm,  CI.FIFTY):.1f} & {coverage(la_dense, dm,  CI.FIFTY):.1f} & {coverage(vi_pred, dm, CI.FIFTY):.1f} & {coverage(mfvi_pred, dm, CI.FIFTY):.1f} & {coverage(hmc_pred, dm, CI.FIFTY):.1f} 
        \bottomrule
        \end{{tabular}}
        }}
    \caption{{Coverage Table}}
\end{{table*}}"""
    with open(target, "w") as f:
        f.write(tab)


def visualize(
    dm,
    ssla_kfac,
    ssla_diag,
    ssla_dense,
    assla_kfac,
    assla_diag,
    assla_dense,
    la_kfac,
    la_diag,
    la_dense,
    vi_pred,
    mfvi_pred,
    hmc_pred,
    target: Path,
):

    fig, axes = plt.subplots(4, 3, figsize=(20, 10))
    axes[0, 0] = uct.plot_xy(
        ssla_kfac["pred"].flatten().numpy(),
        ssla_kfac["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(f"SSLA KFAC")
    axes[0, 0].set_aspect("auto")

    axes[0, 1] = uct.plot_xy(
        ssla_diag["pred"].flatten().numpy(),
        ssla_diag["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(f"SSLA Diag")
    axes[0, 1].set_aspect("auto")

    axes[0, 2] = uct.plot_xy(
        ssla_dense["pred"].flatten().numpy(),
        ssla_dense["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[0, 2],
    )
    axes[0, 2].set_title(f"SSLA Dense")
    axes[0, 2].set_aspect("auto")

    axes[1, 0] = uct.plot_xy(
        assla_kfac["pred"].flatten().numpy(),
        assla_kfac["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[1, 0],
    )
    axes[1, 0].set_title(f"ASSLA KFAC")
    axes[1, 0].set_aspect("auto")

    axes[1, 1] = uct.plot_xy(
        assla_diag["pred"].flatten().numpy(),
        assla_diag["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(f"ASSLA Diag")
    axes[1, 1].set_aspect("auto")

    axes[1, 2] = uct.plot_xy(
        assla_dense["pred"].flatten().numpy(),
        assla_dense["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[1, 2],
    )
    axes[1, 2].set_title(f"ASSLA Dense")
    axes[1, 2].set_aspect("auto")

    axes[2, 0] = uct.plot_xy(
        la_kfac["pred"].flatten().numpy(),
        la_kfac["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[2, 0],
    )
    axes[2, 0].set_title(f"LA KFAC")
    axes[2, 0].set_aspect("auto")

    axes[2, 1] = uct.plot_xy(
        la_diag["pred"].flatten().numpy(),
        la_diag["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[2, 1],
    )
    axes[2, 1].set_title(f"LA Diag")
    axes[2, 1].set_aspect("auto")

    axes[2, 2] = uct.plot_xy(
        la_dense["pred"].flatten().numpy(),
        la_dense["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[2, 2],
    )
    axes[2, 2].set_title(f"LA Dense")
    axes[2, 2].set_aspect("auto")

    axes[3, 0] = uct.plot_xy(
        mfvi_pred["pred"].flatten().numpy(),
        mfvi_pred["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[3, 0],
    )
    axes[3, 0].set_title(f"MFVI")
    axes[3, 0].set_aspect("auto")

    axes[3, 1] = uct.plot_xy(
        vi_pred["pred"].flatten().numpy(),
        vi_pred["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[3, 1],
    )
    axes[3, 1].set_title(f"VI")
    axes[3, 1].set_aspect("auto")

    axes[3, 2] = uct.plot_xy(
        hmc_pred["pred"].flatten().numpy(),
        hmc_pred["pred_uct"].flatten().numpy(),
        dm.Y_test.flatten().numpy(),
        dm.X_test.flatten().numpy(),
        ax=axes[3, 2],
    )
    axes[3, 2].set_title(f"HMC")
    axes[3, 2].set_aspect("auto")

    handles, labels = [], []
    for ax in fig.axes:
        ax.legend().set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", which="major", labelsize=10)
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    handles = [
        mlines.Line2D(
            [], [], color="#ff7f0e", marker="o", linestyle="None", label="Observations"
        ),
        mlines.Line2D([], [], color="#1f77b4", label="Predictions"),
        mpatches.Patch(facecolor="lightsteelblue", alpha=0.4, label="$95\\%$ Interval"),
    ]

    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        ncol=3,
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.subplots_adjust(wspace=0.15, hspace=0.45, top=0.88)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target)
