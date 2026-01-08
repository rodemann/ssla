import json
from pathlib import Path


def convert_json_to_table(
    src,
    target,
    caption: str | None = None,
    label: str | None = None,
    rotate: bool = False,
):
    with open(src, "rb") as f:
        data = json.loads(f.read())
    first_item = data[0]
    no_headers = len(first_item.keys())
    headers = "|c" * no_headers + "|"
    header = " & ".join(first_item.keys())
    from functools import reduce

    header = reduce(
        lambda state, next: state + " & " + rf"{next}", first_item.keys(), ""
    )[2:]
    LINE_END = r"\\ \hline"

    def recursive_format(s: list | str):
        if isinstance(s, list):
            return list(map(recursive_format, s))
        else:
            return s.__format__(".2f")

    lines = "\n".join(
        list(
            map(
                lambda item: reduce(
                    lambda state, next: state + " & " + str(recursive_format(next)),
                    item.values(),
                    "",
                )[2:]
                + LINE_END,
                data,
            )
        )
    )
    if rotate:
        TEMPLATE = rf"""\begin{{table}}[htbp]
        \centering
        \tiny
        \begin{{adjustbox}}{{angle=90}}
            \begin{{tabular}}{{{headers}}}
                \hline
                {header} \\
                \hline
                {lines}
            \end{{tabular}}
        \end{{adjustbox}}
        \caption{{{caption if caption else ''}}}
        \label{{{label if label else ''}}}
    \end{{table}}
    """
    else:
        TEMPLATE = rf"""\begin{{table}}[htbp]
        \centering
        \begin{{tabular}}{{{headers}}}
            \hline
            {header} \\
            \hline
            {lines}
        \end{{tabular}}
        \caption{{{caption if caption else ''}}}
        \label{{{label if label else ''}}}
    \end{{table}}
    """

    with open(target, "w") as f:
        f.write(TEMPLATE)


def convert_states(source: Path, target: Path):
    for file in source.glob("**/*.json"):
        convert_json_to_table(file, target / (file.stem + ".tex"), rotate=True)
