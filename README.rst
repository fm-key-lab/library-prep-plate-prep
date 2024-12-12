.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=======================
library-prep-plate-prep
=======================


    Find optimal plate arrangements.


Find optimal arrangements for sequencing samples on a plate during library prep.

Installation
============

.. code-block:: bash

    git clone https://github.com/username/library-prep-plate-prep.git
    cd library-prep-plate-prep
    pip install -e .

    # or

    pip install git+https://github.com/username/repository.git

Usage
=====

From command line:

.. code-block:: bash

    lppp -i /path/to/samples.csv -o arrangement.csv

In Python:

.. code-block:: python

    import library_prep_plate_prep as lppp

    num_samples = 150
    sample_data = lppp.simulate_data(num_samples)
    print(design.head(2))
    # species  donor  family  timepoint
    # sample                                                          
    # species 4:B003_0001 @ 03 days        4      1       3          3
    # species 3:B001_0001 @ 14 days        3      1       1         14

    # Provide initial values
    plate_init = {
        'n_controls': [3, 2],
        'n_columns': [10, 10],
        'n_rows': [8, 8],
        'n_empty': 2,
    }

    samples = lppp.geometries.SequencingSamples.from_samples(
        sample_data,
        plate_init['n_empty'],
        sum(plate_init['n_controls']),
        cost_fn=lppp.costs.CovarSimilarity()
    )

    # problem init with geometries
    prob = lppp.problems.ArrangementProblem(plates, samples)

    # seed control wells
    ctrls_seeder = lppp.solvers.LHSampler()
    ctrls_arrangement = ctrls_seeder(prob, nt=plate_init['n_controls'])

    # solve arrangement
    solver = lppp.solvers.QAP_2opt()
    soln = solver(prob, partial_match=ctrls_arrangement)

    plate_arrangement = lppp.problems.soln_to_df(prob, soln)
    print(plate_arrangement.head(1))
    # plate  column row well
    # sample                                               
    # species 1:B002_0002 @ 03 days      0       1   A   A1

With custom cost functions:

.. code-block:: python

    custom_sample_cost_fn = lppp.costs.CovarSimilarity.from_rules(
        {
            'species': 10,
            'species_&_family': 0,
            'species_&_donor': 0,
            'species_&_family_&_timepoint': 0,
            'species_&_donor_&_family': 0,
            'species_&_donor_&_family_&_timepoint': 0,
        }
    )

    samples = lppp.geometries.SequencingSamples.from_samples(
        sample_data,
        plate_init['n_empty'],
        sum(plate_init['n_controls']),
        cost_fn=custom_sample_cost_fn
    )

Plotting tools:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 2), layout='constrained')
    lppp.plotting.plate_costs(plates, ncols=5, fig=fig, ax=ax)

.. image:: plate_costfn.png
  :width: 400
  :alt: plate

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), layout='constrained')
    lppp.plotting.sample_costs(samples, ax=ax)

.. image:: xcont_costfn.png
  :width: 400
  :alt: crosscontamination

References
==========

- 2016 Mathematical modeling. 4.3.4 The Transportation Problem
- 2015 Princeton companion mathematics. VI.18 The Traveling Salesman Problem. William Cook
- 2023 Design Heuristics 2.5.4 Quadratic Assignment, 9 Local Search Learning, 9.2 Strategic Oscillations, code listing 9.1

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
