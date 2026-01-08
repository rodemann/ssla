from pathlib import Path


def generate_table(res: list[dict], target: Path):
    entry_template = r"""\multirow{{4}}{{*}}{{{dataset_name}}} & SSLA (cond) & {ssla_cond_95} & {ssla_cond_75} & {ssla_cond_50} & {ssla_cond_nll} & {ssla_cond_crps} \\
 & SSLA & {ssla_95} & {ssla_75} & {ssla_50} & {ssla_nll} & {ssla_crps} \\
 & ASSLA & {assla_95:.2f} & {assla_75:.2f} & {assla_50:.2f} & {assla_nll:.2f} & {assla_crps:.2f} \\
 & LA & {la_95:.2f} & {la_75:.2f} & {la_50:.2f} & {la_nll:.2f} & {la_crps:.2f} \\\hline"""
    entry_template = r"""\multirow{{3}}{{*}}{{{dataset_name}}} & SSLA & {ssla_95} & {ssla_75} & {ssla_50} & {ssla_nll} & {ssla_crps} \\
 & ASSLA & {assla_95:.2f} & {assla_75:.2f} & {assla_50:.2f} & {assla_nll:.2f} & {assla_crps:.2f} \\
 & LA & {la_95:.2f} & {la_75:.2f} & {la_50:.2f} & {la_nll:.2f} & {la_crps:.2f} \\\hline"""
    entries = []
    for dataset in res:
        if dataset["SSLA"] is not None:
            # ssla_cond_95 = dataset["SSLA_COND"]["95"].__format__(".2f")
            # ssla_cond_75 = dataset["SSLA_COND"]["75"].__format__(".2f")
            # ssla_cond_50 = dataset["SSLA_COND"]["50"].__format__(".2f")
            # ssla_cond_nll = dataset["SSLA_COND"]["nll"].__format__(".2f")
            # ssla_cond_crps = dataset["SSLA_COND"]["crps"].__format__(".2f")

            ssla_95 = dataset["SSLA"]["95"].__format__(".2f")
            ssla_75 = dataset["SSLA"]["75"].__format__(".2f")
            ssla_50 = dataset["SSLA"]["50"].__format__(".2f")
            ssla_nll = dataset["SSLA"]["nll"].__format__(".2f")
            ssla_crps = dataset["SSLA"]["crps"].__format__(".2f")
        else:
            # ssla_cond_95 = "-"
            # ssla_cond_75 = "-"
            # ssla_cond_50 = "-"
            # ssla_cond_nll = "-"
            # ssla_cond_crps = "-"

            ssla_95 = "-"
            ssla_75 = "-"
            ssla_50 = "-"
            ssla_nll = "-"
            ssla_crps = "-"
        entry = entry_template.format(
            dataset_name=dataset["name"],
            # ssla_cond_95=ssla_cond_95,
            # ssla_cond_75=ssla_cond_75,
            # ssla_cond_50=ssla_cond_50,
            # ssla_cond_nll=ssla_cond_nll,
            # ssla_cond_crps=ssla_cond_crps,
            ssla_95=ssla_95,
            ssla_75=ssla_75,
            ssla_50=ssla_50,
            ssla_nll=ssla_nll,
            ssla_crps=ssla_crps,
            assla_95=dataset["ASSLA"]["95"],
            assla_75=dataset["ASSLA"]["75"],
            assla_50=dataset["ASSLA"]["50"],
            assla_nll=dataset["ASSLA"]["nll"],
            assla_crps=dataset["ASSLA"]["crps"],
            la_95=dataset["LA"]["95"],
            la_75=dataset["LA"]["75"],
            la_50=dataset["LA"]["50"],
            la_nll=dataset["LA"]["nll"],
            la_crps=dataset["LA"]["crps"],
        )
        entries.append(entry)

    entries = "\n".join(entries)
    table = rf"""\begin{{table}}
\centering
\begin{{tabular}}{{lllllll}}
  &  & \multicolumn{{3}}{{c}}{{Coverage}} & \multicolumn{{2}}{{c}}{{}} \\ 
\cline{{3-5}} 
Dataset & Method & $95\%$ & $75\%$ & $50\%$ & NLL & CRPS \\\hline
 {entries} 
\end{{tabular}}
\caption{{UCIML Results}}
\label{{tab:uciml_results}}
\end{{table}}"""

    target.parent.mkdir(exist_ok=True)
    with open(target, "w") as f:
        f.write(table)
