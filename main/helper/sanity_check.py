import numpy as np
import pandas as pd
from ..offline.greedy import *
from ..offline.zorder import *
from ..offline.states import *
from ..online.counter import *
from ..online.workfunction import *
from ..utils.workload import *
from ..utils.tree import *
import argparse


class Object(object):
    pass


def run_offline():
    # Z ordering
    if args.method == "z":
        z = Zorder(df, config, args.k, 32, queries, False)
        z.make_partitions()
        read = z.eval(queries)
        print("[Uniform Z-order] avg skipped: %.3f" % (1 - read))
        print("[Uniform Z-order] total cost: %f" % (read * len(queries)))
        print()
    # QD-tree
    else:
        tb = TreeBuilder(df, df_sample, config, args, args.k, "", False)
        if not args.load:
            eval_tree = tb.compute_offline_oracle(config["ds"], queries, False)
        else:
            eval_tree = tb.load_offline_oracle(config["ds"], queries)
        read = eval_tree.eval(queries)
        eval_tree.print_splits()
        print("[QD-tree offline] avg skipped: %f" % (1 - read))
        print("[QD-tree offline] total cost: %f" % (read * len(queries)))
        print()


def run_offline_per_wl(workload):
    avg_read = 0
    query_cost = 0
    l = 0
    movement_cost = 0
    for [id, n] in workload:
        r = l + n
        if args.method == 'z':
            z = Zorder(df, config, args.k, 32, queries[l:r], False)
            z.make_partitions()
            read = z.eval(queries[l:r])
        else:
            tb = TreeBuilder(df, df_sample, config, args, args.k, "", False)
            tree = tb.compute_offline_oracle(config["ds"], queries[l:r], False)
            read = tree.eval(queries[l:r])
            tree.save_by_path("%s/%d.p" % (oracle_dir, id))
        avg_read += read
        query_cost += read * len(queries[l:r])
        movement_cost += alpha
        l = r
    avg_read /= args.wl
    method_str = 'Z-order'
    if args.method != 'z':
        method_str = 'QD-tree'
    print("[%s per workload] avg skipped: %f" % (method_str, 1 - avg_read))
    print("[%s per workload] Query: %f, Movement: %f" % (method_str, query_cost, movement_cost))
    print()


def run_random(cm, sg):
    for i in range(T):
        new_queries = queries[i * args.interval:(i + 1) * args.interval]
        new_states = sg.process_queries(new_queries)
        for state in new_states:
            cm.add_state(state)
        cm.process_queries(new_queries)

    print("Total #states: %d" % len(cm.states))
    return cm.query_cost, cm.movement_cost


def run_wfa(sg, wf):
    for i in range(T):
        # Evaluate current tree on new queries
        new_queries = queries[i*args.interval:(i+1)*args.interval]
        wf.process_queries(new_queries)
        # Generate new trees according to policy
        new_trees = sg.process_queries(new_queries)
        wf.add_states(new_trees)
    return wf.query_cost, wf.movement_cost


def point_query_workload():
    q_config = {}
    q_cols = np.random.choice(cols, args.wl, replace=False)
    for i in range(args.wl):
        selected = [q_cols[i]]
        spec = {}
        for col in selected:
            spec[col] = "=="
        q_config[str(i)] = spec

    wl = WorkloadSimulator(len(q_config)).gen_workload(args.queries, not args.sm)
    keys = list(q_config.keys())
    gen = PredicateGenerator(config, config["ds"])
    gen.compute_meta(df)
    preds = []
    for [id, n] in wl:
        spec = q_config[keys[id]]
        queries = gen.gen_pred(spec, n)
        preds.extend(queries)
    workload = []
    out = {}
    for i, pred in enumerate(preds):
        clause = build_predicate(config, pred, 1)
        workload.append(clause)
        out[str(i)] = clause
    pickle.dump(out, open("resources/query/syn-sm.p", "wb"))
    return workload, wl


if __name__ == "__main__":
    cols = ['c' + str(i) for i in range(20)]
    config = {'cat_cols': [],
              'date_cols': [],
              'bool_cols': [],
              'sort_cols': ['c0', 'c1'],
              'num_cols': cols,
              'ds': 'synthetic',
              'delimiter': ',',
              'path': 'na'}

    parser = argparse.ArgumentParser(description='Workload generator.')
    parser.add_argument('--config', default="synthetic", help="Config File Path")
    parser.add_argument('--q', default="random", help="Query Config File Path")
    parser.add_argument('--k', type=int, default=8, help="# partitions")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--policy', default="res")
    parser.add_argument('--queries', type=int, default=5000)
    parser.add_argument('--sm', action='store_true')
    parser.add_argument('--wl', type=int, default=5)
    parser.add_argument('--res', type=int, default=500)
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    # Fix the random seeds for our experiments
    np.random.seed(args.seed)
    df = pd.DataFrame(np.random.randint(0, 10000, size=(50000, 20)),
                      columns=cols)
    sample_size = int(min(max(args.k * 1000, 0.01 * len(df)), len(df)))
    df_sample = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    output_dir = "resources/labels/random/%s-%s-%d" % (
        config["ds"], args.q, args.k)
    oracle_dir = "resources/labels/random/%s-%s-%d-oracle" % (
        config["ds"], args.q, args.k)
    if args.sm:
        output_dir = output_dir + "-sm"
        oracle_dir = oracle_dir + "-sm"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(oracle_dir):
        os.makedirs(oracle_dir)
    queries, workload = point_query_workload()
    T = len(queries) // args.interval
    alpha = args.alpha
    eps = args.eps

    # Sanity check: qd-tree/zordering per workload should be better than
    # building a simple layout for all queries
    for method in ['qd', 'z']:
        args.method = method
        run_offline()
        run_offline_per_wl(workload)

    # Randomized algorithm with best tree for each workload
    args.policy = "oracle"
    args.method = "qd"
    args.load = True
    tb = TreeBuilder(df, df_sample, config, args, args.k, oracle_dir, False)
    init_states = tb.get_init_states()
    query = []
    movement = []
    for _ in range(3):
        cm = CounterManager(tb, init_states, alpha)
        sg = StateGenerator(tb, init_states, args.interval, eps, True)
        sg.reset_reservoir(args.res)
        q, m = run_random(cm, sg)
        query.append(q)
        movement.append(m)
        can_load = True
    print("[Random %s] Query: %f, %f, Movement: %f, %f" %
          (args.policy, np.average(query), np.std(query), np.average(movement), np.std(movement)))

    can_load = False
    args.policy = "res"
    args.load = False
    tb = TreeBuilder(df, df_sample, config, args, args.k, output_dir, False)
    init_states = tb.get_init_states()
    query = []
    movement = []
    for _ in range(3):
        cm = CounterManager(tb, init_states, alpha)
        load = args.load or can_load
        sg = StateGenerator(tb, init_states, args.interval, eps, load)
        sg.reset_reservoir(args.res)
        q, m = run_random(cm, sg)
        query.append(q)
        movement.append(m)
        can_load = True
    print("[Random %s] Query: %f, %f, Movement: %f, %f" %
          (args.policy, np.average(query), np.std(query), np.average(movement), np.std(movement)))

    # WFA
    # sg = StateGenerator(tb, init_states, args.interval, eps, True)
    # wf = WorkFunction(tb, init_states, alpha)
    # q, m = run_wfa(sg, wf)
    # print("[WFA %s] Query: %f, Movement: %f" % (args.policy, q, m))