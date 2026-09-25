"""Reusable matplotlib helpers shared by the figure scripts."""

import numpy as np
import matplotlib.pyplot as plt


def plot_articulatory_parameters(pv,
                                 labels=("J", "B", "D", "T", "LP", "LH", "Hy"),
                                 x_label="Frame index",
                                 y_label="Parameter value",
                                 title="Evolution of articulatory parameters (Pv)",
                                 x_range=None,
                                 y_range=None,
                                 grid=True,
                                 fig_number=None,
                                 ax=None,
                                 new_figure=True,
                                 color=None,
                                 alpha=1.0,
                                 linewidth=1):
    """
    Plot the columns of Pv over time in vertical subplots.

    Can either create a new figure or overlay the curves (dashed) on an
    existing set of axes.

    Args:
        pv (np.ndarray): Array of shape (T, N); each column is a series.
        labels (tuple of str): Names of the N series (subplot y labels).
        x_label (str): Shared x-axis label.
        y_label (str): General y-axis label (kept for API compatibility).
        title (str): Main figure title.
        x_range (tuple or None): (xmin, xmax); defaults to (0, T-1).
        y_range (tuple or None): (ymin, ymax); defaults to (-4, 5).
        grid (bool): Show the grid.
        fig_number (int or None): Matplotlib figure number.
        ax (array-like of Axes or None): Existing axes to overlay on;
            if provided, new_figure must be False.
        new_figure (bool): Create a new figure if True.
        color (str, list, or None): Curve color(s).
        alpha (float): Curve transparency (0 to 1).
        linewidth (float): Line width.

    Returns:
        tuple:
            fig (Figure or None): The created figure (new_figure=True) or
                the figure owning the provided axes.
            axes (list of Axes): The axes used for plotting.
    """
    _, n_dim = pv.shape
    t = np.arange(pv.shape[0])

    # Color handling
    if color is not None:
        if isinstance(color, str):
            colors = [color] * n_dim
        elif hasattr(color, "__len__"):
            colors = color
        else:
            colors = [color] * n_dim
    else:
        colors = [None] * n_dim

    # Overlay on an existing figure
    if not new_figure and ax is not None:
        axes = np.array(ax).flatten()
        if len(axes) != n_dim:
            raise ValueError(
                f"Number of provided axes ({len(axes)}) must match the "
                f"number of parameters ({n_dim})"
            )

        fig = axes[0].figure

        # Draw dashed overlay curves on each existing axis
        for i, (ax_i, col) in enumerate(zip(axes, pv.T)):
            ax_i.plot(t, col, "--", linewidth=linewidth, color=colors[i], alpha=alpha)

        # Only apply limits if specified, to preserve the existing layout
        if x_range is not None:
            for ax_i in axes:
                ax_i.set_xlim(x_range)
        if y_range is not None:
            for ax_i in axes:
                ax_i.set_ylim(y_range)

        plt.draw()

    else:
        # Create a new figure (original behavior)
        if new_figure or ax is None:
            fig, axes = plt.subplots(n_dim, 1, sharex=True, figsize=(8, 2 * n_dim),
                                     num=fig_number)
        else:
            axes = np.array(ax).flatten()
            fig = axes[0].figure

        # Single-parameter case: axes is not a list
        if n_dim == 1:
            axes = [axes] if isinstance(axes, np.ndarray) else [axes]

        # Configure each subplot
        for i, (ax_i, col) in enumerate(zip(axes, pv.T)):
            ax_i.plot(t, col, linewidth=linewidth, color=colors[i], alpha=alpha)

            # Y label with the parameter name
            if i == 0 or new_figure:
                ax_i.set_ylabel(labels[i] if labels and i < len(labels) else f"Param {i}")

            # Axis limits
            xmin, xmax = x_range if x_range is not None else (0, pv.shape[0] - 1)
            ymin, ymax = y_range if y_range is not None else (-4, 5)
            ax_i.set_xlim(xmin, xmax)
            ax_i.set_ylim(ymin, ymax)

            if grid and (new_figure or i == 0):
                ax_i.grid(True, alpha=0.3)

        # Shared axis configuration
        axes[-1].set_xlabel(x_label)

        # Main title (new figures only)
        if new_figure:
            fig.suptitle(title, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)  # Leave room for the title

        # Display (new figures only)
        if new_figure:
            plt.show(block=False)

    return fig, axes


def plot_ctw_path(path_matrix, ref_label="Reference index", target_label="Target index",
                  title="CTW path", figsize=(6, 5), show_grid=True, marker=".",
                  linestyle="-", color="gray", alpha=1.0, ax=None):
    """
    Plot a CTW warping path between two sequences.

    Args:
        path_matrix (np.ndarray): Array of shape (n_points, 2); each row
            contains (idx_ref, idx_tar).
        ref_label (str): X-axis label.
        target_label (str): Y-axis label.
        title (str): Plot title.
        figsize (tuple): Figure size (if ax is None).
        show_grid (bool): Show a grid.
        marker (str): Scatter marker style.
        linestyle (str): Line style.
        color (str): Plot color (default gray).
        alpha (float): Transparency.
        ax (matplotlib.axes.Axes or None): Existing axes; if None, a new
            figure is created.

    Returns:
        matplotlib.axes.Axes: The axes used.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    idx_ref = path_matrix[:, 0]
    idx_tar = path_matrix[:, 1]

    # Thin line with small points (dense paths)
    ax.plot(idx_ref, idx_tar, linestyle=linestyle, color=color, alpha=alpha,
            linewidth=0.1)
    ax.scatter(idx_ref, idx_tar, marker=marker, color=color, alpha=alpha,
               s=0.1, zorder=5)

    ax.set_xlabel(ref_label)
    ax.set_ylabel(target_label)
    ax.set_title(title)
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_xlim(min(idx_ref) - 1, max(idx_ref) + 1)
    ax.set_ylim(min(idx_tar) - 1, max(idx_tar) + 1)

    return ax
