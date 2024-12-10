from typing import Optional, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from library_prep_plate_prep import geometries

__all__ = ['plate_costs', 'sample_costs']


def plate_costs(
    plate: Union[geometries.Plate, geometries.Plates],
    cmap: Optional[str] = 'Greys',
    ncols: Optional[int] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    
    if not (isinstance(plate, geometries.Plate) or isinstance(plate, geometries.Plates)):
        raise ValueError
    elif isinstance(plate, geometries.Plates):
        nc = plate._plates[0].columns
        nr = plate._plates[0].rows
    else:
        nc = plate.columns
        nr = plate.rows

    nc = nc if ncols is None else ncols

    plate_costs_subset = plate.cost_matrix[:nr, :(nr * nc)]

    def add_col_outline(ax) -> None:
        okws = {'lw': 1, 'ec': 'k', 'fc': 'none'}
        def outline_col(i):
            rect = Rectangle((-.5+nr*i, -.5), nr, nr, **okws)
            ax.add_patch(rect)
        [outline_col(_) for _ in range(nc)]

    def add_x_eq_y(ax) -> None:
        ylim = ax.get_ylim()
        ax.plot(ylim[::-1], ylim[::-1], color='k', ls='--')

    hmap_kws = {'aspect': 'equal', 'vmin': 0, 'cmap': cmap}
    hmap = ax.imshow(plate_costs_subset, **hmap_kws)

    def add_cbar(ax) -> None:
        max_cost = np.floor(plate_costs_subset.max())
        cax = ax.inset_axes([1.04, 0.2, 0.02, 0.6])
        fig.colorbar(hmap, cax=cax, extend='max', ticks=[0, max_cost])
        cax.set_yticklabels(['Near', 'Far']);
    
    for plt_f in [add_col_outline, add_x_eq_y, add_cbar]:
        plt_f(ax)

    ax.set_xlabel('Well Labels', size=10)
    ax.set_ylabel('Well Labels\n(Column 1)', size=10)

    WELL_LABELS = np.arange(nc*nr) # TODO: address with a plate mixin
    ax.set_xticks(np.arange(nc*nr), WELL_LABELS, size=7);
    ax.set_yticks(np.arange(nr), WELL_LABELS[:nr], size=8);

    ax.set_title('"Plate distance" costs', size=12)


def sample_costs(
    samples: geometries.SequencingSamples,
    cmap: Optional[dict[str]] = {'family': 'Set1', 'timepoint': 'viridis', 'cost': 'Reds'},
    nonsamples: Optional[list[str]] = ['control', 'empty'],
    ax: Optional[Axes] = None,
):
    
    design = (
        samples._data
        .copy(deep=True)
        .reset_index()
        .query("index not in @nonsamples")
        .sort_values(['family', 'timepoint'])
    )
    
    def plot_marginals():

        def add_marginal(var, iax_dims, dim, odim) -> None:
            label_opts = {'x': 'labeltop', 'y': 'labelright'}
            iax = ax.inset_axes(iax_dims, **{f'share{dim}': ax})
            vals = design[[var]]
            vals = vals.T if dim == 'x' else vals
            iax.imshow(vals.values, aspect='auto', cmap=cmap[var])
            iax.tick_params(axis=dim, **{label_opts[dim]: False})
            iax.set(**{f'{odim}ticks': [0], f'{odim}ticklabels': [var] if dim == 'x' else []})

        iax_pad = -.1
        iax_width = .04
        
        x0, y0, width, height = 0, iax_pad, 1, iax_width
        for dim in ['x', 'y']:
            odim = 'y' if dim == 'x' else 'x'
            for var in ['timepoint', 'family']:
                add_marginal(var, [x0, y0, width, height], dim, odim)
                if dim == 'x':
                    y0 -= .05
                else:
                    x0 -= .05
            xtmp, ytmp, wtmp, htmp = y0 + .05, x0, height, width
            x0, y0, width, height = xtmp, ytmp, wtmp, htmp

    samples_costs_subset = (
        -1 * samples.cost_matrix
        [np.ix_(design.index.values, design.index.values)]
    )

    def plot_main():
        min_cost = np.floor(samples_costs_subset.min())
        max_cost = np.floor(samples_costs_subset.max())
        hmap_kws = {'aspect': 'equal', 'vmin': min_cost, 'vmax': max_cost, 'cmap': 'Reds'}
        
        ccpens = ax.imshow(samples_costs_subset, **hmap_kws)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('"Cross-contamination" costs', size=12)
        
    plot_marginals()
    plot_main()