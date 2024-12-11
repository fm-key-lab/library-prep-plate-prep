import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog='library-prep-plate-prep',
        description='Use transport optimization to solve library prep plate arrangements.',
        epilog='See also https://github.com/t-silvers/library-prep-plate-prep'
    )

    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output', default='arrangement.csv')

    return parser.parse_args()


def run():
    import matplotlib.pyplot as plt
    import pandas as pd

    from library_prep_plate_prep import (
        geometries,
        problems,
        solvers,
        utils,
    )

    # TODO: Add logging
    args = parse_args()

    print(f'Reading input file {args.input} ...')

    # TODO: validate
    data = (
        pd.read_csv(args.input)
        .set_index('sample')
    )

    print('Setting up arrangement problem ...')
    
    pinit = utils.calc_init_vals(data.shape[0])
    prob = problems.ArrangementProblem(
        geometries.Plates(pinit['n_columns'], pinit['n_rows']),
        geometries.SequencingSamples.from_samples(data, pinit['n_empty'], sum(pinit['n_controls']))
    )

    print('Solving problem with default parameters ...')
    
    seeder, solver = solvers.LHSampler(), solvers.QAP_2opt()
    seed = seeder(prob, nt=pinit['n_controls'])
    soln = solver(prob, partial_match=seed)

    print(f'Saving optimal arrangement to {args.output} ...')
    
    (
        problems.soln_to_df(prob, soln)
        .to_csv(args.output)
    )


if __name__ == '__main__':
    run()