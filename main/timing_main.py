import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from e2e.partition import *
import pickle
import argparse
from utils.setup import *
import time
import os
import pickle

#os.environ['PYSPARK_SUBMIT_ARGS'] = \
#    '--packages com.amazonaws:aws-java-sdk:1.7.4,org.apache.hadoop:hadoop-aws:2.7.3 pyspark-shell'

if __name__ == "__main__":
    root = "/mnt/1/partition/test"
    session = SparkSession.builder.config(
        "spark.master", "local[4]").config("spark.driver.memory", "4g").getOrCreate()

    # conf = SparkConf().set('spark.executor.extraJavaOptions', '-Dcom.amazonaws.services.s3.enableV4 = true'). \
    #         set('spark.driver.extraJavaOptions', '-Dcom.amazonaws.services.s3.enableV4 = true'). \
    #         set("spark.driver.memory", "4g"). \
    #         setAppName('pyspark_aws').setMaster('local[4]')
    # sc = SparkContext(conf=conf)
    # sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')
    #
    # accessKeyId ='ASIAZ33ZR7FDTNORH73R'
    # secretAccessKey = 'SmgwE4ELjRcW8zzya4g2mVUQdVw3eqrGP1eixYAd'
    # hadoopConf = sc._jsc.hadoopConfiguration()
    # hadoopConf.set('fs.s3a.access.key', accessKeyId)
    # hadoopConf.set('fs.s3a.secret.key', secretAccessKey)
    # hadoopConf.set('fs.s3a.endpoint', 's3-us-west-1.amazonaws.com')
    # hadoopConf.set('fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
    #
    # session = SparkSession(sc)
    # df = session.read.csv("/mnt/1/datasets/test.csv")
    #
    # t0 = time.time()
    # df.write.format('parquet').option('header', True).save('s3://reorg-test/test', mode='overwrite')
    # t1 = time.time()
    # print("s3 time: %f" % (t1 - t0))
    #
    # t0 = time.time()
    # df.write.format('parquet').option('header', True).save('/mnt/1/partition/test',
    #        mode='overwrite')
    # t1 = time.time()
    # print("local time: %f" % (t1 - t0))

    # Create base layout
    k = 800
    N = 35772405

    pm = PartitionManager(session, k)
    data_dir = '/mnt/1/datasets/s10z1'
    in_dir = '%s/base' % root
    out_dir = '%s/out' % root

    if False:
        bids = np.random.randint(0, k, N)
        pm.load_and_reorg_with_labels(bids, data_dir, in_dir, True)

    # result = pickle.load(open("results/local_alpha.p", "rb"))
    # query = result["query"]
    # reorg = result["reorg"]
    result = {}
    query = []
    reorg = []
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        print("n=%d" % n)
        parts = list(range(n))
        q = []
        r = []
        # Cold run
        if n == 1:
            pm.run_query(in_dir, parts, "SELECT SUM(L_EXTENDEDPRICE), COUNT(*) from tbl")
            df = pm.load_parquet_bids(in_dir, parts)
            bids = np.random.randint(0, k, N)
            pm.reorg_with_labels(df, bids, out_dir)
        for i in range(3):
            t0 = time.time()
            pm.run_query(in_dir, parts, "SELECT SUM(L_EXTENDEDPRICE), COUNT(*) from tbl")
            t1 = time.time()
            print("query", t1 - t0)
            q.append(t1 - t0)

            t0 = time.time()
            df = pm.load_parquet_bids(in_dir, parts)
            bids = np.random.randint(0, k, N)
            pm.reorg_with_labels(df, bids, out_dir)
            t1 = time.time()
            print("reorg", t1 - t0)
            r.append(t1 - t0)
        query.append([np.average(q), np.std(q)])
        reorg.append([np.average(r), np.std(r)])
        print(query[-1], reorg[-1])

    result["query"] = query
    result["reorg"] = reorg
    print(result)
    pickle.dump(result, open("results/local_alpha4.p", "wb"))
