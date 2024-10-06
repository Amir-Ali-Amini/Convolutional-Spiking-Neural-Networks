import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def format_parameters(inp_parameters, num_columns=4):
    """
    Formats a list of strings into a table-like string with specified number of columns.

    Args:
    parameters (list of str): List of parameter strings.
    num_columns (int): Number of columns in the table.

    Returns:
    str: Formatted table-like string.
    """
    max_len = max(len(param) for param in inp_parameters) + 2  # +2 for padding
    if max_len > 30:
        num_columns = 2

    parameters = inp_parameters + [
        ""
        for _ in range(
            num_columns - ((len(inp_parameters) % num_columns) or num_columns)
        )
    ]

    rows = (len(parameters) + num_columns - 1) // num_columns  # ceiling division
    table_str = ""

    for r in range(rows):
        row_params = parameters[r::rows]  # Get every nth element starting from r
        row_str = " | ".join(param.ljust(max_len) for param in row_params)
        table_str += f" {row_str} \n"

    border_len = len(table_str.split("\n")[0]) - 1
    table_str = table_str

    return table_str


def plot_grid(
    data,
    f,
    plot_titles=[],
    force_height=0,
    title="Extracted Features",
    size=1.5,
    parameters=[],
    scaling_factor=1.5,
    label_font_size=8,
    num_columns=4,
):
    fig_height = force_height or int(f ** (1 / 2))
    fig_width = int(f // fig_height) * 2
    fig_height *= 2
    if len(parameters):
        fig_height += 1
    print(f"height: {fig_height}, width: {fig_width}")
    fig = plt.figure(figsize=(fig_width * size, fig_height * size))
    gs = GridSpec(fig_height, fig_width, figure=fig)
    if len(parameters):
        # ax = fig.add_subplot(fig_height, 1, fig_height)
        ax = fig.add_subplot(gs[fig_height - 1 : fig_height, 0:fig_width])
        ax.axis("off")
        params_text = format_parameters(parameters, num_columns)
        ax.text(
            0.5,
            0.5,
            params_text,
            fontsize=(label_font_size - 3) * scaling_factor,
            verticalalignment="center",
            horizontalalignment="center",
            transform=ax.transAxes,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    j = 0
    w = fig_width // 2
    for i in data:
        # print(
        #     h,
        #     int(j // h) * 2,
        #     int(j // h) * 2 + 2,
        #     int(j % h) * 2,
        #     int(j % h) * 2 + 2,
        # )
        # ax = fig.add_subplot(fig_height + 0.5, fig_width, j + 1)
        ax = fig.add_subplot(
            gs[
                int(j // w) * 2 : int(j // w) * 2 + 2,
                int(j % w) * 2 : int(j % w) * 2 + 2,
            ]
        )

        ax.imshow(i, cmap="gray")
        if len(plot_titles):
            ax.set_title(f"{plot_titles[j]}")
        else:
            ax.set_title(f"feature {j+1}")
        ax.axis("off")
        j += 1

    fig.suptitle(title, fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.show()
