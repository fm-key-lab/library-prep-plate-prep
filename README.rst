.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=======================
library-prep-plate-prep
=======================


    Add a short description here!


A longer description of your project goes here...

Usage
=====

.. code-block:: python

    from library_prep_plate_prep.arranger import PlateArranger
    from library_prep_plate_prep.simulate import simulate_data

    design = simulate_data()
    print(design.head())
    #         donor  timepoint  family
    # sample                          
    # 0           8          4       6
    # 1           4          3       9

    plate_arranger = PlateArranger()
    plate_arranger.arrange(design, controls_per_plate=0)
    print(plate_arranger.soln)
    # solution

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
