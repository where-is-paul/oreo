from ..utils.predicates import *
from ..utils.workload import *
import argparse
import json

template = {
        "q1": "SELECT L_RETURNFLAG, L_LINESTATUS, SUM(L_QUANTITY), SUM(L_EXTENDEDPRICE), " +
              "SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)), SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)*(1+L_TAX)), " +
              "AVG(L_QUANTITY), AVG(L_EXTENDEDPRICE), AVG(L_DISCOUNT), COUNT(*) FROM tbl WHERE %s" +
              " GROUP BY L_RETURNFLAG, L_LINESTATUS ORDER BY L_RETURNFLAG, L_LINESTATUS",
        "q3": "SELECT SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)) as rev, L_ORDERKEY, O_ORDERDATE, O_SHIPPRIORITY" +
              " FROM tbl WHERE %s GROUP BY L_ORDERKEY, O_ORDERDATE, O_SHIPPRIORITY" +
              " ORDER BY rev desc, O_ORDERDATE, L_ORDERKEY",
        "q4": "SELECT O_ORDERPRIORITY, COUNT(*) FROM tbl WHERE %s " +
              " GROUP BY O_ORDERPRIORITY ORDER BY O_ORDERPRIORITY",
        "q5": "SELECT N1_NAME, SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)) as rev FROM tbl WHERE %s " +
              " GROUP BY N1_NAME ORDER BY rev desc",
        "q6": "SELECT SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)) as rev FROM tbl WHERE %s",
        "q7": "SELECT N1_NAME, N2_NAME, YEAR(L_SHIPDATE) as L_YEAR, SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)) as rev" +
              " FROM tbl WHERE %s GROUP BY N1_NAME, N2_NAME, L_YEAR ORDER BY L_YEAR",
        "q8": "SELECT YEAR(O_ORDERDATE) as O_YEAR, SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)) as rev  " +
              " FROM tbl WHERE %s GROUP BY O_YEAR ORDER BY O_YEAR",
        "q10": "SELECT C_CUSTKEY, SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)) as rev" +
              " FROM tbl WHERE %s GROUP BY C_CUSTKEY ORDER BY rev desc, C_CUSTKEY LIMIT 20",
        "q12": "SELECT L_SHIPMODE," +
               " SUM(CASE WHEN O_ORDERPRIORITY = '1-URGENT' OR O_ORDERPRIORITY = '2-HIGH' THEN 1 ELSE 0 END), " +
               " SUM(CASE WHEN O_ORDERPRIORITY <> '1-URGENT' OR O_ORDERPRIORITY <> '2-HIGH' THEN 1 ELSE 0 END)"
               " FROM tbl WHERE %s GROUP BY L_SHIPMODE ORDER BY L_SHIPMODE",
        "q14": "SELECT 100.0 * SUM(CASE WHEN P_TYPE LIKE 'PROMO%' THEN L_EXTENDEDPRICE*(1-L_DISCOUNT) ELSE 0 END)" +
               " / SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)) FROM tbl WHERE ",
        "q17": "SELECT SUM(L_EXTENDEDPRICE)/7. FROM tbl WHERE %s AND L_QUANTITY < (SELECT 0.2 * AVG(L_QUANTITY) FROM tbl)",
        "q19": "SELECT SUM(L_EXTENDEDPRICE*(1-L_DISCOUNT)) FROM tbl WHERE %s",
        "q21": "SELECT S_NAME, COUNT(*) AS c FROM tbl AS t1 WHERE %s AND L_RECEIPTDATE > L_RECEIPTDATE AND " +
               " EXISTS (SELECT * FROM tbl AS t2 WHERE t1.L_ORDERKEY = t2.L_ORDERKEY AND t1.L_SUPPKEY <> t2.L_SUPPKEY)" +
               " AND NOT EXISTS (SELECT * FROM tbl AS t3 WHERE t1.L_ORDERKEY = t3.L_ORDERKEY  " +
               " AND t3.L_SUPPKEY <> t3.L_SUPPKEY AND t3.L_RECEIPTDATE > t3.L_RECEIPTDATE)" +
                " GROUP BY S_NAME ORDER BY c desc, S_NAME LIMIT 100"
    }

ds_template = {
    "q13": """
    select avg(ss_quantity)
       ,avg(ss_ext_sales_price)
       ,avg(ss_ext_wholesale_cost)
       ,sum(ss_ext_wholesale_cost)
    from tbl where %
    """,
    "q19": "select i_brand_id, i_brand, i_manufact_id, i_manufact, sum(ss_ext_sales_price) as ext_price from tbl" +
           " where %s group by i_brand_id, i_brand, i_manufact_id, i_manufact order by ext_price limit 100",
    "q27": "select i_item_id, s_state, avg(ss_quantity), avg(ss_list_price)," +
           "avg(ss_coupon_amt), avg(ss_sales_price) from tbl where %s group by i_item_id, s_state",
    "q28": "select avg(ss_list_price),count(ss_list_price),count(distinct ss_list_price) from tbl where %s"
}

def write_tpch(n, keys, queries, infile):
    f = open("resources/query/%s.sql" % infile, 'w')
    q_idx = 0
    for i, k in enumerate(keys):
        for j in range(n[i]):
            if k == 'q14':
                query = template[k] + str(queries[str(q_idx)])
            else:
                query = template[k] % str(queries[str(q_idx)])
            f.write('%s\n' % (query))
            q_idx += 1
    f.close()

def write_tpcds(n, keys, queries, infile):
    f = open("resources/query/%s.sql" % infile, 'w')
    q_idx = 0
    for i, k in enumerate(keys):
        for j in range(n[i]):
            query = ds_template[k] % str(queries[str(q_idx)])
            f.write('%s\n' % (query))
            q_idx += 1
    f.close()

def write_default(queries, infile):
    template = "SELECT COUNT(*) FROM tbl WHERE %s"
    f = open("resources/query/%s.sql" % infile, 'w')
    cnts = []
    for i in range(len(queries)):
        q = queries[str(i)]
        cnts.append(len(q.get_leaves()))
        f.write("%s\n" % (template % str(q)))
    f.close()
    print("Avg #preds: %f, min: %d, max: %d" % (np.average(cnts), min(cnts), max(cnts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workload generator.')
    parser.add_argument('--infile', default='single')

    args = parser.parse_args()
    queries = pickle.load(open("resources/query/%s.p" % args.infile, "rb"))
    print(len(queries))

    if args.infile == "tpch1":
        n =[731, 2152, 2007, 2710,  572, 1529,  691,  683,  812,  539, 2421, 1973, 2628,
             2235, 799, 2559, 2090,  245, 1403, 1221]
        keys = ['q3', 'q14', 'q19', 'q7', 'q19', 'q14', 'q17', 'q21', 'q17', 'q5', 'q17', 'q12', 'q12',
                'q10', 'q6', 'q4', 'q19', 'q7', 'q8', 'q1']
        write_tpch(n, keys, queries, args.infile)
    elif args.infile == 'tpcds3':
        n = [2166, 2330, 2073,  509, 2607, 1011, 1033, 2507,  295, 1837, 2362, 1322, 2181,
             549, 486, 1415, 1731,  918, 1386, 1282]
        keys = ['q19', 'q53', 'q7', 'q88', 'q3', 'q13', 'q27', 'q48', 'q96', 'q28', 'q52', 'q36', 'q98', 'q79', 'q55',
                'q68', 'q34', 'q89', 'q42', 'q46']
        write_tpcds(n, keys, queries, args.infile)
    else:
        write_default(queries, args.infile)

    with open("resources/config/chunk.json", mode="r") as f:
        config = json.load(f)
    df, _ = load_df(config)
    N = len(df)
    print("#rows: %d" % N)
    read = []
    for i in range(len(queries)):
        q = queries[str(i)]
        read.append(len(df.query(str(q))))
        print(str(q), read[-1])
        if i > 10:
            break
    print(np.average(read) * 1. / N)