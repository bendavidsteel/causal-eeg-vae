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
    db_table_in = 'reuters_news'

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    #mongo_source = sources.MongoDB(db_connection_string, db_name, db_table_in, partition_size_mb=64)
    mongo_source = sources.CSV('/root/google_protest_news.csv')
    collector = collect.Collector(mongo_source)

    graph_constructor = graphs.Graph(nl_processor)
    graph_constructor.build_entity_dag()

    #runner = run.Runner(graph_constructor, master_url=master_url, num_executors=2, executor_cores=22, executor_memory='240g', driver_memory='160g', spark_conf=spark_conf)
    runner = run.Runner(graph_constructor)
    G, map_df = runner.get_obj()
    
    graph_path = os.path.join(data_path, 'all_articles.edgelist')
    map_path = os.path.join(data_path, 'all_article_nodes.map')

    nx.write_weighted_edgelist(G, graph_path)
    map_df.to_csv(map_path, index=False, sep=' ')

if __name__ == '__main__':
    main()