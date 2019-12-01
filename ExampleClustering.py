from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.cure import cure;
from pyclustering.utils import read_sample;
from pyclustering.samples.definitions import FCPS_SAMPLES;
# Input data in following format [ [0.1, 0.5], [0.3, 0.1], ... ].
input_data = read_sample(FCPS_SAMPLES.SAMPLE_LSUN);
# Allocate three clusters.
cure_instance = cure(input_data, 3);
cure_instance.process();
clusters = cure_instance.get_clusters();
# Visualize allocated clusters.
visualizer = cluster_visualizer();
visualizer.append_clusters(clusters, input_data);
visualizer.show();