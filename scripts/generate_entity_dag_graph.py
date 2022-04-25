import datetime
import os

import networkx as nx
from seldonite import sources, collect, nlp, graphs, run

def main():
    master_url = 'k8s://https://10.140.16.25:6443'
    db_connection_string = os.environ['MONGO_CONNECTION_STRING']

    spark_conf = {
        'spark.kubernetes.authenticate.driver.serviceAccountName': 'ben-dev',
        'spark.kubernetes.driver.pod.name': 'seldonite-driver',
        'spark.driver.host': 'seldonite-driver',
        'spark.driver.port': '7078',
        'spark.kubernetes.executor.volumes.hostPath.fake-nfs-mount.mount.path': '/root',
        'spark.kubernetes.executor.volumes.hostPath.fake-nfs-mount.options.path': '/var/nfs/spark'
    }
        #'spark.kubernetes.executor.volumes.nfs.shared-nfs-mount.mount.path': '/root',
        #'spark.kubernetes.executor.volumes.nfs.shared-nfs-mount.options.server': '10.100.90.152',
        #'spark.kubernetes.executor.volumes.nfs.shared-nfs-mount.options.path': '/var/nfs/spark'
    #}

    db_name = 'political_events'
    db_table_in = 'reuters_news_reduced'

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    mongo_source = sources.news.MongoDB(db_connection_string, db_name, db_table_in, partition_size_mb=16)
    #mongo_source = sources.news.CSV('/root/google_protest_news.csv')


    start_date = datetime.date(2007, 1, 1)
    end_date = datetime.date(2016, 1, 1) - datetime.timedelta(days=1)
    collector = collect.Collector(mongo_source) \
        .in_date_range(start_date, end_date)

    nl_processor = nlp.NLP(collector) \
        .get_entities(blacklist_entities=['.*reuters.*', 'free', 'register'], max_string_search=1000)

    graph_constructor = graphs.Graph(nl_processor) \
        .build_entity_dag()

    #runner = run.Runner(graph_constructor, master_url=master_url, num_executors=2, executor_cores=22, executor_memory='64g', driver_memory='64g', spark_conf=spark_conf)
    runner = run.Runner(graph_constructor, driver_cores=24, driver_memory='64g')
    nodes, edges = runner.to_pandas()
    
    node_path = os.path.join(data_path, '0715_entity_dag_nodes.csv')
    edge_path = os.path.join(data_path, '0715_entity_dag_edges.csv')

    nodes.to_csv(node_path, index=False)
    edges.to_csv(edge_path, index=False)

if __name__ == '__main__':
    main()