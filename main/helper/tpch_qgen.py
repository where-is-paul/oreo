from ..utils.predicates import *
from ..utils.workload import *
from ..offline.zorder import *
import numpy as np
from dateutil.relativedelta import relativedelta
from dateutil import parser
import datetime
import argparse
import json


def dt_to_str(date):
    return datetime.datetime.strftime(date, date_fmt)


def random_brand():
    m = np.random.randint(1, 6)
    n = np.random.randint(1, 6)
    return "Brand#%d%d" % (m, n)


def q19_clause(cfg, brand, containers, sizes, q):
    p1 = {"P_BRAND": [brand]}
    p2 = {"P_CONTAINER": containers}
    p3 = {"L_QUANTITY": [q, q+10]}
    p4 = {"P_SIZE": sizes}
    p5 = {"L_SHIPMODE": ['AIR', 'AIR REG']}
    p6 = {"L_SHIPINSTRUCT": ['DELIVER IN PERSON']}
    return build_predicate(cfg, [p1, p2, p3, p4, p5, p6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TPCH query generator.')
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    N = args.n
    with open("../resources/config/denorms1.json", mode="r") as f:
        cfg = json.load(f)
    date_fmt = "%Y-%m-%d"
    gen = PredicateGenerator(cfg, "denorms1", True)

    all_queries = {}
    # Q1: l_shipdate <= '1998-12-01' - [60~120] units day
    end = datetime.datetime(1998,12,1)
    end_date = "1998-12-01"
    queries = []
    for i in range(N):
        n = -1 * np.random.randint(60, 121)
        delta = relativedelta(days=n)
        start = end + delta
        pred = Expr(cfg, {"L_SHIPDATE": [np.nan, dt_to_str(start)]})
        queries.append(pred)
    all_queries["q1"] = queries

    # Q3: c_mktsegment = 'BUILDING' and o_orderdate < MDY(3, 15, 1995)
    #     AND l_shipdate > MDY(3, 15, 1995)
    queries = []
    for i in range(N):
        day = np.random.randint(1, 32)
        date = datetime.datetime(1995, 3, day)
        p1 = gen._cat_query("C_MKTSEGMENT", 1)
        p2 = {"O_ORDERDATE": [np.nan, dt_to_str(date)]}
        p3 = {"L_SHIPDATE": [dt_to_str(date), np.nan]}
        pred = build_predicate(cfg, [p1, p2, p3])
        queries.append(pred)
    all_queries["q3"] = queries


    # Q4: o_orderdate >= date ':1'
    # 	and o_orderdate < date ':1' + interval '3' month
    queries = []
    for i in range(N):
        m = np.random.randint(0, 58)
        start = datetime.datetime(1993, 1, 1) + relativedelta(months=m)
        end = start + relativedelta(months=3)
        pred = Expr(cfg, {"O_ORDERDATE": [
            dt_to_str(start), dt_to_str(end)]})
        queries.append(pred)
    all_queries["q4"] = queries


    # Q5: r_name = ':1'
    # 	and o_orderdate >= date ':2'
    # 	and o_orderdate < date ':2' + interval '1' year
    queries = []
    for i in range(N):
        p1 = gen._cat_query("R1_NAME", 1)
        y = np.random.randint(1993, 1998)
        start = datetime.datetime(y, 1, 1)
        end = start + relativedelta(years=1)
        p2 = {"O_ORDERDATE": [dt_to_str(start), dt_to_str(end)]}
        pred = build_predicate(cfg, [p1, p2])
        queries.append(pred)
    all_queries["q5"] = queries

    # Q6: l_shipdate >= date ':1'
    # 	and l_shipdate < date ':1' + interval '1' year
    # 	and l_discount between :2 - 0.01 and :2 + 0.01
    # 	and l_quantity < :3;
    queries = []
    for i in range(N):
        y = np.random.randint(1993, 1998)
        start = datetime.datetime(y, 1, 1)
        end = start + relativedelta(years=1)
        p1 = {"L_SHIPDATE": [dt_to_str(start), dt_to_str(end)]}
        dis = np.random.randint(2, 10) / 100.
        p2 = {"L_DISCOUNT": [dis, dis+0.01]}
        qty = np.random.randint(24, 26)
        p3 = {"L_QUANTITY": [np.nan, qty]}
        pred = build_predicate(cfg, [p1, p2, p3])
        queries.append(pred)
    all_queries["q6"] = queries

    # Q7: ( (n1.n_name = ':1' and n2.n_name = ':2')
    # 		or (n1.n_name = ':2' and n2.n_name = ':1') )
    #       and l_shipdate between date '1995-01-01' and date '1996-12-31'
    queries = []
    for i in range(N):
        p1 = {"L_SHIPDATE": ['1995-01-01', '1996-12-31']}
        v1 = gen._cat_query("N1_NAME", 1)["N1_NAME"]
        v2 = gen._cat_query("N2_NAME", 1)["N2_NAME"]
        tmp1 = build_predicate(cfg, [{"N1_NAME": v1}, {"N2_NAME": v2}], 1)
        tmp2 = build_predicate(cfg, [{"N1_NAME": v2}, {"N2_NAME": v1}], 1)
        left = Predicate(tmp1, tmp2, LogicalOp.OR)
        right = Expr(cfg, p1)
        pred = Predicate(left, right, LogicalOp.AND)
        queries.append(pred)
    all_queries["q7"] = queries

    # Q8: r_name = ':2'
    # 		and o_orderdate between date '1995-01-01' and date '1996-12-31'
    # 		and p_type = ':3'
    queries = []
    for i in range(N):
        p1 = {"O_ORDERDATE": ['1995-01-01', '1996-12-31']}
        p2 = gen._cat_query("P_TYPE", 1)
        p3 = gen._cat_query("R1_NAME", 1)
        pred = build_predicate(cfg, [p1, p2, p3])
        queries.append(pred)
    all_queries["q8"] = queries

    # Q9: p_name like '%:1%'
    key = 'q9,%d'
    # p_name = random.choice(colors, 5)
    # too many distinct values to filter
    colors = ["almond",  "antique",  "aquamarine",  "azure",  "beige",  "bisque",  "black",  "blanched",  "blue", "blush",
              "brown",  "burlywood",  "burnished",  "chartreuse",  "chiffon",  "chocolate",  "coral", "cornflower",
              "cornsilk",  "cream",  "cyan",  "dark",  "deep",  "dim",  "dodger",  "drab",  "firebrick", "floral",
              "forest",  "frosted",  "gainsboro",  "ghost",  "goldenrod",  "green",  "grey",  "honeydew", "hot",
              "indian",  "ivory",  "khaki",  "lace",  "lavender",  "lawn",  "lemon",  "light",  "lime",  "linen",
              "magenta",  "maroon",  "medium",  "metallic",  "midnight",  "mint",  "misty",  "moccasin",  "navajo",
              "navy",  "olive",  "orange",  "orchid",  "pale",  "papaya",  "peach",  "peru",  "pink",  "plum",
              "powder", "puff",  "purple",  "red",  "rose",  "rosy",  "royal",  "saddle",  "salmon",  "sandy",
              "seashell",  "sienna", "sky",  "slate",  "smoke",  "snow",  "spring",  "steel",  "tan",  "thistle",
              "tomato",  "turquoise",  "violet", "wheat",  "white",  "yellow"]

    # Q10: o_orderdate >= date ':1'
	#   and o_orderdate < date ':1' + interval '3' month
    #   and l_returnflag = 'R'
    queries = []
    for i in range(N):
        mo = np.random.randint(0, 24)
        start = datetime.datetime(1993,2,1) + relativedelta(months=mo)
        end = start + relativedelta(months=3)
        p1 = {"O_ORDERDATE": [dt_to_str(start), dt_to_str(end)]}
        p2 = {"L_RETURNFLAG": ["R"]}
        pred = build_predicate(cfg, [p1, p2])
        queries.append(pred)
    all_queries["q10"] = queries

    # Q12: l_shipmode in (':1', ':2')
    # 	and l_receiptdate >= date ':3'
    # 	and l_receiptdate < date ':3' + interval '1' year
    queries = []
    for i in range(N):
        y = np.random.randint(1993, 1998)
        start = datetime.datetime(y, 1, 1)
        end = start + relativedelta(years=1)
        p1 = {"L_RECEIPTDATE": [dt_to_str(start), dt_to_str(end)]}
        p2 = gen._cat_query("L_SHIPMODE", 2)
        pred = build_predicate(cfg, [p1, p2])
        queries.append(pred)
    all_queries["q12"] = queries

    # Q14: l_shipdate >= date ':1'
    # 	and l_shipdate < date ':1' + interval '1' month;
    queries = []
    for i in range(N):
        y = np.random.randint(1993, 1998)
        mo = np.random.randint(1,13)
        start = datetime.datetime(y, mo, 1)
        end = start + relativedelta(months=1)
        p1 = {"L_SHIPDATE": [dt_to_str(start), dt_to_str(end)]}
        pred = Expr(cfg, p1)
        queries.append(pred)
    all_queries["q14"] = queries

    # Q17: p_brand = ':1'
    # 	and p_container = ':2'
    queries = []
    for i in range(N):
        p1 = {"P_BRAND": [random_brand()]}
        p2 = gen._cat_query("P_CONTAINER", 1)
        pred = build_predicate(cfg, [p1, p2])
        queries.append(pred)
    all_queries["q17"] = queries

    # Q18 filter on sum(l_quantity) group by l_orderkey

    # Q19: (
    # 		p_brand = ':1'
    # 		and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
    # 		and l_quantity >= :4 and l_quantity <= :4 + 10
    # 		and p_size between 1 and 5
    # 		and l_shipmode in ('AIR', 'AIR REG')
    # 		and l_shipinstruct = 'DELIVER IN PERSON'
    # 	)
    # 	or
    # 	(
    # 		p_brand = ':2'
    # 		and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
    # 		and l_quantity >= :5 and l_quantity <= :5 + 10
    # 		and p_size between 1 and 10
    # 		and l_shipmode in ('AIR', 'AIR REG')
    # 		and l_shipinstruct = 'DELIVER IN PERSON'
    # 	)
    # 	or
    # 	(
    # 		p_brand = ':3'
    # 		and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
    # 		and l_quantity >= :6 and l_quantity <= :6 + 10
    # 		and p_size between 1 and 15
    # 		and l_shipmode in ('AIR', 'AIR REG')
    # 		and l_shipinstruct = 'DELIVER IN PERSON'
    # 	);
    queries = []
    for i in range(N):
        q1 = np.random.randint(1, 11)
        q2 = np.random.randint(10, 21)
        q3 = np.random.randint(20, 31)
        b1 = random_brand()
        b2 = random_brand()
        b3 = random_brand()
        c1 = q19_clause(cfg, b1, ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'], [1, 5], q1)
        c2 = q19_clause(cfg, b2, ['MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'], [1, 10], q2)
        c3 = q19_clause(cfg, b3, ['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'], [1, 15], q2)
        tmp = Predicate(c1, c2, LogicalOp.OR)
        pred = Predicate(tmp, c3, LogicalOp.OR)
        queries.append(pred)
    all_queries["q19"] = queries

    # Q21: o_orderstatus = 'F' and n_name = ':1'
    queries = []
    for i in range(N):
        p1 = {"O_ORDERSTATUS": ["F"]}
        p2 = gen._cat_query("N2_NAME", 1)
        pred = build_predicate(cfg, [p1, p2])
        queries.append(pred)
    all_queries["q21"] = queries
    pickle.dump(all_queries, open("resources/query/tpch-oracle.p", "wb"))

    # N = 30000
    # n_wl = 20
    # min_size = 200
    # prob = np.random.uniform(0, 1, n_wl)
    # prob /= np.sum(prob)
    # wl = np.random.multinomial(N-n_wl*min_size, prob, 1)[0] + min_size
    # print(wl)
    # workload = {}
    # idx = 0
    # keys = list(all_queries.keys())[::-1]
    # random.shuffle(keys)
    # keys.extend(np.random.choice(list(all_queries.keys()), n_wl - len(all_queries)))
    # random.shuffle(keys)
    # print(keys, len(set(keys)))
    # for i, n in enumerate(wl):
    #     k = keys[i]
    #     for j in range(n):
    #         workload[str(idx)] = all_queries[k][j]
    #         idx += 1
    # print(len(workload))
    #
    # with open("resources/config/denorm.json", mode="r") as f:
    #     config = json.load(f)
    # df = load_csv("/lfs/1/krong/datasets/denorm-sample/denorm10.csv", config)
    # print(len(df))
    # z = Zorder(df, config, 40, 32)
    # z.make_partitions()
    # read, _ = z.eval(list(workload.values()))
    # print("Average skipped: %f" % (1 - read))
    # pickle.dump(workload, open("resources/query/tpch1.p", "wb"))

















