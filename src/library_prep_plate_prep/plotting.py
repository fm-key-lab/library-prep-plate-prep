import matplotlib.pyplot as plt
import numpy as np

from .inference import plate_distance_matrix
from .plates import Plate


# TODO: temporary, should replace plate with or add X
def plot_plate_distances(plate: Plate, fig, ax):
    
    # TODO: If fig, ax not provided    
    # fig, ax = plt.subplots(figsize=(9, 2), layout='constrained')

    num_columns, num_rows = plate.size
    num_wells = num_columns * num_rows
    well_labels = plate.data.coords['well'].values
    
    D = plate_distance_matrix(plate)

    dists = ax.imshow(D[:num_rows, :], aspect='equal', vmin=0, cmap='Greys')

    box_kwargs = dict(lw=1, ec='k', fc='none')

    for col in range(num_columns):
        ax.add_patch(
            plt.Rectangle(
                (-.5 + num_rows * col, -.5), num_rows, num_rows, **box_kwargs
            )
        )

    ax.plot(
        ax.get_ylim()[::-1],
        ax.get_ylim()[::-1],
        color='k',
        ls='--'
    )

    ax.set_xlabel('Well Labels', size=10)

    ax.set_xticks(
        np.arange(num_wells),
        well_labels.flatten(),
        size=7,
    )

    ax.set_ylabel('Well Labels\n(Column 1)', size=10)

    ax.set_yticks(
        np.arange(num_rows),
        well_labels.flatten()[:num_rows],
        size=8,
    )

    ax.set_title('Distance between wells', size=12)

    cax = ax.inset_axes([1.04, 0.2, 0.02, 0.6])
    fig.colorbar(dists, cax=cax, extend='max', ticks=[0, np.floor(D.max())])
    cax.set_yticklabels(['Near', 'Far'])


def plot_sample_penalties(cost_matrix, design, fig, scale=1):    
    
    ax = fig.add_gridspec(bottom=0.75, left=0.75).subplots()
    ax.set(aspect=1)

    # Marginals

    ax_x1 = ax.inset_axes([0, -.2, 1, 0.2], sharex=ax)
    ax_x1.imshow(design[::scale, 1].T.reshape(1, -1), aspect='equal', cmap='viridis')
    ax_x1.tick_params(axis='x', labeltop=False)
    ax_x1.set_yticks([0], ['Date'])

    ax_x2 = ax.inset_axes([0, -.25, 1, 0.15], sharex=ax)
    ax_x2.imshow(design[::scale, 2].T.reshape(1, -1), aspect='equal', cmap='Set1')
    ax_x2.tick_params(axis='x', labeltop=False)
    ax_x2.set_yticks([0], ['Family'])

    ax_y1 = ax.inset_axes([-.2, 0, 0.2, 1], sharey=ax)
    ax_y1.imshow(design[::scale, 1].reshape(-1, 1), aspect='equal', cmap='viridis')
    ax_y1.tick_params(axis='y', labelright=False)
    ax_y1.set_xticks([])

    ax_y2 = ax.inset_axes([-.25, 0, 0.15, 1], sharey=ax)
    ax_y2.imshow(design[::scale, 2].reshape(-1, 1), aspect='equal', cmap='Set1')
    ax_y2.tick_params(axis='y', labelright=False)
    ax_y2.set_xticks([])

    # Main

    # Remove controls/empties and down-sample
    cost_matrix_sampled = cost_matrix[::scale, ::scale]

    ccpens = ax.imshow(cost_matrix_sampled, aspect='equal', vmin=0, vmax=10, cmap='Reds')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Cross-contamination penalties', size=12)