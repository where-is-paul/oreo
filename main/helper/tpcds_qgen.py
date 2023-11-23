from ..utils.predicates import *
from ..utils.workload import *
from ..offline.zorder import *
import numpy as np
from dateutil.relativedelta import relativedelta
from dateutil import parser
import datetime
import argparse
import json

dvs = {
    "gender": ["M", "F"],
    "marital_status": ["S", "U", "M", "W", "D"],
    "education": ["Unknown", "Primary", "Secondary", "College", "Advanced Degree", "4 yr Degree", "2 yr Degree"],
    "state": ["KS","CA","NH","OR","ND","TX","NV","KY","OH","NY","HI","NM","IN","MS","DC","WV","NE","FL","MO","AR","ME",
              "CT","WI","NC","SD","RI","OK","ID","GA","MN","PA","MD","AK","WY","LA","MT","IL","TN","NJ","MI","WA","MA",
              "AL","IA","UT","VT","CO","SC","VA","DE","AZ"],
    "i_cat": ["Electronics","Sports","Books","Home","Children","Music","Men","Jewelry", "Women","Shoes"],
    "i_class": ["earings","tables","memory","kids","womens","disk drives","womens watch","home repair","bedding","athletic","parenting","fiction","stereo","bathroom","portable","sports-apparel","basketball","flatware","reference","decor","school-uniforms","camcorders","rock","camping","gold","pendants","personal","lighting","accessories","custom","outdoor","arts","dresses","diamonds","sailing","glassware","science","hockey","wireless","curtains/drapes","accent","golf","fitness","archery","rings","maternity","infants","football","mens","bracelets","monitors","dvd/vcr players","self-help","scanners","blinds/shades","romance","audio","guns","furniture","wallpaper","birdal","paint","mystery","televisions","computers","classical","fishing","cooking","travel","business","swimwear","pop","costume","athletic shoes","history","consignment","entertainments","musical","mens watch","rugs","newborn","baseball","automotive","optics","karoke","tennis","estate","pools","shirts","jewelry boxes","loose stones","cameras","toddlers","mattresses","sports","pants","country","semi-precious","fragrances"],
    "s_city": ["Five Points", "Pleasant Hill", "Midway", "Fairview","Riverside", "Oak Grove"],
    "buy": ["0-500","Unknown","1001-5000","501-1000",">10000","5001-10000"],
    "s_county": ["Ziebach County", "Williamson County", "Walker County"]

}

def dt_to_str(date):
    return datetime.datetime.strftime(date, date_fmt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TPCH query generator.')
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', default='tpcds0')
    args = parser.parse_args()

    N = args.n
    with open("../resources/config/ss.json", mode="r") as f:
        cfg = json.load(f)
    date_fmt = cfg["date_format"]

    all_queries = {}
    # Q13
    # MS = ulist(dist(marital_status, 1, 1), 3);
    # ES = ulist(dist(education, 1, 1), 3);
    # STATE = ulist(dist(fips_county, 3, 1), 9);
    queries = []
    for i in range(N):
        p0 = {"d_year": [2001]}

        p1 = {"cd_marital_status": list(np.random.choice(dvs["marital_status"], 1))}
        p2 = {"cd_education_status": list(np.random.choice(dvs["education"], 1))}
        p3 = {"ss_sales_price": [100.00, 150.00]}
        p4 = {"hd_dep_count": [3]}
        c1 = build_predicate(cfg, [p1, p2, p3, p4])
        p1 = {"cd_marital_status": list(np.random.choice(dvs["marital_status"], 1))}
        p2 = {"cd_education_status": list(np.random.choice(dvs["education"], 1))}
        p3 = {"ss_sales_price": [50.00, 100.00]}
        p4 = {"hd_dep_count": [1]}
        c2 = build_predicate(cfg, [p1, p2, p3, p4])
        p1 = {"cd_marital_status": list(np.random.choice(dvs["marital_status"], 1))}
        p2 = {"cd_education_status": list(np.random.choice(dvs["education"], 1))}
        p3 = {"ss_sales_price": [150.00, 200.00]}
        p4 = {"hd_dep_count": [1]}
        c3 = build_predicate(cfg, [p1, p2, p3, p4])
        or1 = build_predicate(cfg, [c1, c2, c3], 0)

        p1 = {"ca_country": ["United States"]}
        p2 = {"ca_state": list(np.random.choice(dvs["state"], 3, replace=False))}
        p3 = {"ss_net_profit": [100, 200]}
        c1 = build_predicate(cfg, [p1, p2, p3])
        p1 = {"ca_country": ["United States"]}
        p2 = {"ca_state": list(np.random.choice(dvs["state"], 3, replace=False))}
        p3 = {"ss_net_profit": [150, 300]}
        c2 = build_predicate(cfg, [p1, p2, p3])
        p1 = {"ca_country": ["United States"]}
        p2 = {"ca_state": list(np.random.choice(dvs["state"], 3, replace=False))}
        p3 = {"ss_net_profit": [50, 250]}
        or2 = build_predicate(cfg, [c1, c2, c3], 0)

        pred = build_predicate(cfg, [p0, or1, or2])
        queries.append(pred)
    all_queries["q13"] = queries

    # Q19
    # YEAR = random(1998, 2002, uniform);
    # MONTH = random(11, 12, uniform);
    # MGR_IDX = dist(i_manager_id, 1, 1);
    # MANAGER = random(distmember(i_manager_id, [MGR_IDX], 2), distmember(i_manager_id, [MGR_IDX], 3), uniform);
    # _LIMIT = 100;
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2003)
        m = np.random.randint(11, 13)
        m_id = np.random.randint(1, 101)
        p1 = {"d_year": [y]}
        p2 = {"d_moy": [m]}
        p3 = {"i_manager_id": [m_id]}
        pred = build_predicate(cfg, [p1, p2, p3])
        queries.append(pred)
    all_queries["q19"] = queries

    # Q27
    # YEAR = random(1998, 2002, uniform);
    # GEN = dist(gender, 1, 1);
    # MS = dist(marital_status, 1, 1);
    # ES = dist(education, 1, 1);
    # STATENUMBER = ulist(random(1, rowcount("active_states", "store"), uniform), 6);
    # STATE_A = distmember(fips_county, [STATENUMBER.1], 3);
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2003)
        gen = list(np.random.choice(dvs["gender"], 1))
        ms = list(np.random.choice(dvs["marital_status"], 1))
        es = list(np.random.choice(dvs["education"], 1))
        states = list(np.random.choice(dvs["state"], 6, replace=False))
        p1 = {"cd_gender": gen}
        p2 = {"cd_marital_status": ms}
        p3 = {"cd_education_status": es}
        p4 = {"d_year": [y]}
        p5 = {"s_state": states}
        pred = build_predicate(cfg, [p1, p2, p3, p4, p5])
        queries.append(pred)
    all_queries["q27"] = queries

    # Q28
    # LISTPRICE = ulist(random(0, 190, uniform), 6);
    # COUPONAMT = ulist(random(0, 18000, uniform), 6);
    # WHOLESALECOST = ulist(random(0, 80, uniform), 6);
    queries = []
    for i in range(N):
        coup = np.random.uniform(0, 18000)
        ws = np.random.uniform(0, 80)
        p1 = {"ss_quantity": [0, 5]}
        p2 = {"ss_list_price": sorted(np.random.uniform(0, 190, 2))}
        p3 = {"ss_coupon_amt": [coup, coup+1000]}
        p4 = {"ss_wholesale_cost": [ws, ws+20]}
        p5 = build_predicate(cfg, [p1, p2])
        c1 = build_predicate(cfg, [p3, p4, p5], 0)

        coup = np.random.uniform(0, 18000)
        ws = np.random.uniform(0, 80)
        p1 = {"ss_quantity": [6, 10]}
        p2 = {"ss_list_price": sorted(np.random.uniform(0, 190, 2))}
        p3 = {"ss_coupon_amt": [coup, coup + 1000]}
        p4 = {"ss_wholesale_cost": [ws, ws + 20]}
        p5 = build_predicate(cfg, [p1, p2])
        c2 = build_predicate(cfg, [p3, p4, p5], 0)

        coup = np.random.uniform(0, 18000)
        ws = np.random.uniform(0, 80)
        p1 = {"ss_quantity": [11, 15]}
        p2 = {"ss_list_price": sorted(np.random.uniform(0, 190, 2))}
        p3 = {"ss_coupon_amt": [coup, coup + 1000]}
        p4 = {"ss_wholesale_cost": [ws, ws + 20]}
        p5 = build_predicate(cfg, [p1, p2])
        c3 = build_predicate(cfg, [p3, p4, p5], 0)

        coup = np.random.uniform(0, 18000)
        ws = np.random.uniform(0, 80)
        p1 = {"ss_quantity": [16, 20]}
        p2 = {"ss_list_price": sorted(np.random.uniform(0, 190, 2))}
        p3 = {"ss_coupon_amt": [coup, coup + 1000]}
        p4 = {"ss_wholesale_cost": [ws, ws + 20]}
        p5 = build_predicate(cfg, [p1, p2])
        c4 = build_predicate(cfg, [p3, p4, p5], 0)

        coup = np.random.uniform(0, 18000)
        ws = np.random.uniform(0, 80)
        p1 = {"ss_quantity": [21, 25]}
        p2 = {"ss_list_price": sorted(np.random.uniform(0, 190, 2))}
        p3 = {"ss_coupon_amt": [coup, coup + 1000]}
        p4 = {"ss_wholesale_cost": [ws, ws + 20]}
        p5 = build_predicate(cfg, [p1, p2])
        c5 = build_predicate(cfg, [p3, p4, p5], 0)

        coup = np.random.uniform(0, 18000)
        ws = np.random.uniform(0, 80)
        p1 = {"ss_quantity": [26, 30]}
        p2 = {"ss_list_price": sorted(np.random.uniform(0, 190, 2))}
        p3 = {"ss_coupon_amt": [coup, coup + 1000]}
        p4 = {"ss_wholesale_cost": [ws, ws + 20]}
        p5 = build_predicate(cfg, [p1, p2])
        c6 = build_predicate(cfg, [p3, p4, p5], 0)

        pred = build_predicate(cfg, [c1, c2, c3, c4, c5, c6], 0)
        queries.append(pred)
    all_queries["q28"] = queries

    # Q3
    # define MONTH = random(11,12,uniform);
    # define MANUFACT= random(1,1000,uniform);
    queries = []
    for i in range(N):
        m = np.random.randint(11, 13)
        m_id = np.random.randint(1, 1001)
        p1 = {"d_moy": [m]}
        p2 = {"i_manufact_id": [m_id]}
        pred = build_predicate(cfg, [p1, p2])
        queries.append(pred)
    all_queries["q3"] = queries

    # Q34
    # YEAR = random(1998, 2000, uniform);
    # BPONE = text({"1001-5000", 1}, {">10000", 1}, {"501-1000", 1});
    # BPTWO = text({"0-500", 1}, {"Unknown", 1}, {"5001-10000", 1});
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2001)
        p1 = {"d_dom": [1,3]}
        p2 = {"d_dom": [25,28]}
        p3 = {"hd_buy_potential": list(np.random.choice(dvs["buy"], 2, replace=False))}
        p4 = {"hd_vehicle_count": [1, np.nan]}
        p5 = {"d_year": [y, y+1, y+2]}
        p6 = {"s_county": list(np.random.choice(dvs["s_county"], 2, replace=False))}
        c0 = build_predicate(cfg, [p1, p2], 0)
        pred = build_predicate(cfg, [c0, p3, p4, p5, p6])
        queries.append(pred)
    all_queries["q34"] = queries

    # Q36
    #  define YEAR=random(1998,2002,uniform);
    #  define STATENUMBER=ulist(random(1, rowcount("active_states", "store"), uniform),8);
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2003)
        states = list(np.random.choice(dvs["state"], 6, replace=False))
        p1 = {"d_year": [y]}
        p2 = {"s_state": states}
        pred = build_predicate(cfg, [p1, p2])
        queries.append(pred)
    all_queries["q36"] = queries

    # Q42
    # MONTH = random(11, 12, uniform); YEAR = random(1998, 2002, uniform);
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2003)
        m = np.random.randint(11, 13)
        p1 = {"d_year": [y]}
        p2 = {"d_moy": [m]}
        p3 = {"i_manager_id": [1]}
        pred = build_predicate(cfg, [p1, p2, p3])
        queries.append(pred)
    #all_queries["q42"] = queries

    # Q46
    # DEPCNT = random(0, 9, uniform);
    # YEAR = random(1998, 2000, uniform);
    # VEHCNT = random(-1, 4, uniform);
    # CITYNUMBER = ulist(random(1, rowcount("active_cities", "store"), uniform), 5);
    queries = []
    for i in range(N):
        dep = np.random.randint(0, 10)
        year = np.random.randint(1998,2001)
        vc = np.random.randint(-1, 5)
        cities = list(np.random.choice(dvs["s_city"], 5, replace=False))
        p1 = {"hd_dep_count": [dep]}
        p2 = {"hd_vehicle_count": [vc]}
        p3 = {"d_dow": [0, 6]}
        p4 = {"d_year": [year, year+1,year+2]}
        p5 = {"s_city": cities}
        c1 = build_predicate(cfg, [p1, p2], 0)
        pred = build_predicate(cfg, [p3, p4, p5, c1])
        queries.append(pred)
    all_queries["q46"] = queries

    # Q48
    # MS = ulist(dist(marital_status, 1, 1), 3);
    # ES = ulist(dist(education, 1, 1), 3);
    # STATE = ulist(dist(fips_county, 3, 1), 9);
    # YEAR = random(1998, 2002, uniform);
    queries = []
    for i in range(N):
        p0 = {"d_year": [np.random.randint(1998, 2003)]}

        p1 = {"cd_marital_status": list(np.random.choice(dvs["marital_status"], 1))}
        p2 = {"cd_education_status": list(np.random.choice(dvs["education"], 1))}
        p3 = {"ss_sales_price": [100.00, 150.00]}
        c1 = build_predicate(cfg, [p1, p2, p3])

        p1 = {"cd_marital_status": list(np.random.choice(dvs["marital_status"], 1))}
        p2 = {"cd_education_status": list(np.random.choice(dvs["education"], 1))}
        p3 = {"ss_sales_price": [50.00, 100.00]}
        c2 = build_predicate(cfg, [p1, p2, p3])

        p1 = {"cd_marital_status": list(np.random.choice(dvs["marital_status"], 1))}
        p2 = {"cd_education_status": list(np.random.choice(dvs["education"], 1))}
        p3 = {"ss_sales_price": [150.00, 200.00]}
        c3 = build_predicate(cfg, [p1, p2, p3])

        p1 = {"ca_state": list(np.random.choice(dvs["state"], 3, replace=False))}
        p2 = {"ss_net_profit": [0, 2000]}
        c4 = build_predicate(cfg, [p1, p2])

        p1 = {"ca_state": list(np.random.choice(dvs["state"], 3, replace=False))}
        p2 = {"ss_net_profit": [150, 3000]}
        c5 = build_predicate(cfg, [p1, p2])

        p1 = {"ca_state": list(np.random.choice(dvs["state"], 3, replace=False))}
        p2 = {"ss_net_profit": [50, 25000]}
        c6 = build_predicate(cfg, [p1, p2])

        or1 = build_predicate(cfg, [c1, c2, c3], 0)
        or2 = build_predicate(cfg, [c4, c5, c6], 0)
        pred = build_predicate(cfg, [or1, or2])
        queries.append(pred)
    all_queries["q48"] = queries

    # Q52
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2003)
        m = np.random.randint(1, 13)
        p1 = {"d_year": [y]}
        p2 = {"d_moy": [m]}
        p3 = {"i_manager_id": [1]}
        pred = build_predicate(cfg, [p1, p2, p3])
        queries.append(pred)
    #all_queries["q52"] = queries

    # Q53
    queries = []
    for i in range(N):
        dms = np.random.randint(1176, 1225)
        p1 = {"d_month_seq": list(range(dms, dms+12))}
        p2 = {"i_category": ['Books','Children','Electronics']}
        p3 = {"i_class": ['personal','portable','reference','self-help']}
        p4 = {"i_brand": ['scholaramalgamalg #14','scholaramalgamalg #7',
		    'exportiunivamalg #9','scholaramalgamalg #9']}
        c1 = build_predicate(cfg, [p2, p3, p4])
        p5 = {"i_category": ['Women','Music','Men']}
        p6 = {"i_class": ['accessories','classical','fragrances','pants']}
        p7 = {"i_brand": ['amalgimporto #1','edu packscholar #1','exportiimporto #1',
		    'importoamalg #1']}
        c2 = build_predicate(cfg, [p5, p6, p7])
        or1 = build_predicate(cfg, [c1, c2], 0)
        pred = build_predicate(cfg, [or1, p1])
        queries.append(pred)
    all_queries["q53"] = queries

    # Q55
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2003)
        m = np.random.randint(1, 13)
        m_id = np.random.randint(1, 101)
        p1 = {"d_year": [y]}
        p2 = {"d_moy": [m]}
        p3 = {"i_manager_id": [m_id]}
        pred = build_predicate(cfg, [p1, p2, p3])
        queries.append(pred)
    #all_queries["q55"] = queries

    # Q68
    queries = []
    for i in range(N):
        dep = np.random.randint(0, 10)
        year = np.random.randint(1998, 2001)
        vc = np.random.randint(-1, 5)
        cities = list(np.random.choice(dvs["s_city"], 2, replace=False))
        p0 = {"d_dom":[1, 2]}
        p1 = {"hd_dep_count": [dep]}
        p2 = {"hd_vehicle_count": [vc]}
        p3 = {"d_year": [year, year + 1, year + 2]}
        p4 = {"s_city": cities}
        c1 = build_predicate(cfg, [p1, p2], 0)
        pred = build_predicate(cfg, [p3, p4, p0, c1])
        queries.append(pred)
    all_queries["q68"] = queries

    # Q7
    # GEN = dist(gender, 1, 1);
    # MS = dist(marital_status, 1, 1);
    # ES = dist(education, 1, 1);
    # YEAR = random(1998, 2002, uniform);
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2003)
        gen = list(np.random.choice(dvs["gender"], 1))
        ms = list(np.random.choice(dvs["marital_status"], 1))
        es = list(np.random.choice(dvs["education"], 1))
        p1 = {"cd_gender": gen}
        p2 = {"cd_marital_status": ms}
        p3 = {"cd_education_status": es}
        p4 = {"d_year": [y]}
        p5 = {"p_channel_email": ["N"]}
        p6 = {"p_channel_event": ["N"]}
        c1 = build_predicate(cfg, [p5, p6], 0)
        pred = build_predicate(cfg, [p1, p2, p3, p4, c1])
        queries.append(pred)
    all_queries["q7"] = queries

    # Q79
    #  define DEPCNT=random(0,9,uniform);
    #  define YEAR = random(1998,2000,uniform);
    #  define VEHCNT=random(-1,4,uniform);
    queries = []
    for i in range(N):
        dep = np.random.randint(0, 10)
        year = np.random.randint(1998, 2001)
        vc = np.random.randint(-1, 5)
        cities = list(np.random.choice(dvs["s_city"], 5, replace=False))
        p1 = {"hd_dep_count": [dep]}
        p2 = {"hd_vehicle_count": [vc, np.nan]}
        p3 = {"d_dow": [0, 6]}
        p4 = {"d_year": [year, year + 1, year + 2]}
        p5 = {"s_number_employees": [200, 295]}
        c1 = build_predicate(cfg, [p1, p2], 0)
        pred = build_predicate(cfg, [p3, p4, p5, c1])
        queries.append(pred)
    all_queries["q79"] = queries

    # Q88
    # HOUR = ulist(random(-1, 4, uniform), 3);
    # STORE = dist(stores, 1, 1);
    queries = []
    for i in range(N):
        cnts = np.random.randint(-1, 5, 3)
        unions = []
        for h in range(8, 13):
            for j in range(2):
                if h == 8 and j == 0 or h == 12 and j == 1:
                    continue
                p0 = {"s_store_name": ["ese"]}
                p1 = {"t_hour": [h]}
                if j == 0:
                    p2 = {"t_minute": [np.nan, 30]}
                else:
                    p2 = {"t_minute": [30, np.nan]}
                cs = []
                for i in range(3):
                    p3 = {"hd_dep_count": [cnts[i]]}
                    p4 = {"hd_vehicle_count": [np.nan, cnts[i]+2]}
                    cs.append(build_predicate(cfg, [p3, p4]))
                c1 = build_predicate(cfg, cs)
                c2 = build_predicate(cfg, [p0, p1, p2])
                unions.append(build_predicate(cfg, [c1, c2]))
        pred = build_predicate(cfg, unions)
        queries.append(pred)
    all_queries["q88"] = queries

    # Q89
    # YEAR = random(1998, 2002, uniform);
    # IDX = ulist(random(1, rowcount("categories"), uniform), 6);
    # CAT_A = distmember(categories, [IDX.1], 1);
    # CLASS_A = DIST(distmember(categories, [IDX.1], 2), 1, 1);
    queries = []
    for i in range(N):
        p0 = {"d_year": [np.random.randint(1998, 2003)]}
        p1 = {"i_category": list(np.random.choice(dvs["i_cat"], 3, replace=False))}
        p2 = {"i_category": list(np.random.choice(dvs["i_cat"], 3, replace=False))}
        #p3 = {"i_class": np.random.choice(dvs["i_class"], 3, replace=False)}
        #p4 = {"i_class": np.random.choice(dvs["i_class"], 3, replace=False)}
        c1 = build_predicate(cfg, [p1, p2], 0)
        pred = build_predicate(cfg, [p0, c1])
        queries.append(pred)
    all_queries["q89"] = queries

    # Q96
    queries = []
    for i in range(N):
        p0 = {"s_store_name": ["ese"]}
        p1 = {"t_hour": list(np.random.choice([8, 16, 15, 20],1))}
        p2 = {"t_minute": [30, np.nan]}
        p3 = {"hd_dep_count": [np.random.randint(0, 10)]}
        pred = build_predicate(cfg, [p0, p1, p2, p3])
        queries.append(pred)
    all_queries["q96"] = queries

    # Q98
    # YEAR = random(1998, 2002, uniform);
    # SDATE = date([YEAR] + "-01-01", [YEAR] + "-07-01", sales);
    # CATEGORY = ulist(dist(categories, 1, 1), 3);
    queries = []
    for i in range(N):
        y = np.random.randint(1998, 2003)
        cats = list(np.random.choice(dvs["i_cat"], 3, replace=False))
        p1 = {"i_category": cats}
        start = datetime.datetime(y,1,1) + relativedelta(days=np.random.randint(1, 182))
        end = start + relativedelta(days=30)
        p2 = {"d_date": [dt_to_str(start), dt_to_str(end)]}
        preds = build_predicate(cfg, [p1, p2])
        queries.append(pred)
    all_queries["q98"] = queries
    pickle.dump(all_queries, open("resources/query/tpcds-oracle.p", "wb"))

    # N = 30000
    # n_wl = 20
    # min_size = 200
    # all_keys = list(all_queries.keys())
    # print(len(all_keys))
    # keys = list(all_keys)
    # keys.extend(list(np.random.choice(all_keys, n_wl - len(all_keys), replace=False)))
    # random.shuffle(keys)
    # prob = np.random.uniform(0, 1, n_wl)
    # prob /= np.sum(prob)
    # wl = np.random.multinomial(N - n_wl * min_size, prob, 1)[0] + min_size
    # print(wl, keys)
    # workload = {}
    # idx = 0
    # for i, n in enumerate(wl):
    #     k = keys[i]
    #     for j in range(n):
    #         workload[str(idx)] = all_queries[k][j]
    #         idx += 1
    #
    # print(len(workload))
    # df = load_csv("/lfs/1/krong/datasets/ss/combined.csv", cfg)
    # df = df.sample(frac=0.1)
    # z = Zorder(df, cfg, 10, 32)
    # z.make_partitions()
    #
    # read, _ = z.eval(list(workload.values()))
    # print("Average skipped: %f" % (1 - read))
    # pickle.dump(workload, open("resources/query/%s.p" % args.out, "wb"))





