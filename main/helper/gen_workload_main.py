from ..utils.predicates import *
from ..utils.workload import *
from ..offline.zorder import *
import numpy as np
from dateutil.relativedelta import relativedelta
from dateutil import parser
import datetime
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workload generator.')
    parser.add_argument('--config', help="Config File Path")
    parser.add_argument('--q', default="pickle", help="Query Config File Path")
    parser.add_argument('--k', type=int, default=100, help="# partitions")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--queries', type=int, default=20000)

    # Real dataset
    args = parser.parse_args()
    # Fix the random seeds for our experiments
    np.random.seed(args.seed)

    with open("resources/config/%s.json" % args.config, mode="r") as f:
        config = json.load(f)
    with open("resources/query/%s.json" % args.q, mode="r") as f:
        q_config = json.load(f)
    df, fname = load_df(config)
    sample_size = int(min(max(args.k * 10000, 0.001 * len(df)), len(df)))
    df_sample = df.sample(n=sample_size, random_state=args.seed)
    gen = PredicateGenerator(config, fname, True)
    #gen.compute_meta(df)

    wl = WorkloadSimulator(len(q_config)).gen_workload(args.queries, False)
    keys = list(q_config.keys())
    print(wl)
    cnt = {}
    preds = []
    for [id, n] in wl:
        if not keys[id] in cnt:
            cnt[keys[id]] = 0
        cnt[keys[id]] += n
        spec = q_config[keys[id]]
        queries = gen.gen_pred(spec, n)
        preds.extend(queries)
    print(len(cnt), cnt)

    workload = {}
    for i, pred in enumerate(preds):
        clause = build_predicate(config, pred, 1)
        workload[str(i)] = clause

    pickle.dump(workload, open("resources/query/%s-sm%d.p" % (args.config,args.seed), "wb"))

    # Estimate selectivity
    z = Zorder(df_sample, config, 50, 32)
    z.make_partitions()
    read, _ = z.eval(list(workload.values()))
    print("Average skipped: %f" % (1 - read))
