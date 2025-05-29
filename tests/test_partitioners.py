import unittest
import random
from tests import util
from main.partitioner.metis_cython import MetisCythonPartitioner
from main.partitioner.metis_wrapper import MetisPartitioner

# (n, p, num_partitions, mp)
parameters = [
    (100, 0.3, 5, 0),
    (100, 0.3, 10, 0),
    (100, 0.3, 10, 1),
    (1000, 0.5, 10, 3)
]

class BasePartitionerTest:
    partitioner_class = None
    
    def test_all_nodes_in_a_partition(self):
        """
        The union of all partitions should equal the set of nodes of the graph
        """
        
        random.seed(42)
        
        partitioner = self.partitioner_class()
        
        for n, p, num_partitions, mp in parameters:
            with self.subTest(f"(n, p, num_partitions, mp) = ({n}, {p}, {num_partitions}, {mp})"):
                adjlist = util.erdos_renyi_adjlist(n, p)
                
                core, extended = partitioner.partition(None, adjlist, num_partitions, mp)

                for i in range(100):
                    contained = False
                    for part in core:
                        if i in part:
                            contained = True
                            break
                        
                    self.assertTrue(contained)
            
    def test_core_partitions_disjoint(self):
        """
        No node should be contained in two core partitions
        """
        
        random.seed(42)
                
        partitioner = self.partitioner_class()
        
        for n, p, num_partitions, mp in parameters:
            with self.subTest(f"(n, p, num_partitions, mp) = ({n}, {p}, {num_partitions}, {mp})"):
                adjlist = util.erdos_renyi_adjlist(n, p)

                core, extended = partitioner.partition(None, adjlist, num_partitions, mp)

                for i in range(100):
                    contained = False
                    for part in core:
                        if i in part:
                            if contained:
                                self.fail(f'Node #{i} is duplicated')
                            contained = True
                        
                    self.assertTrue(contained)
                    
    def test_neighbor_inclusion(self):
        """
        No node should be contained in two core partitions
        """
        
        random.seed(42)
                
        partitioner = self.partitioner_class()
        
        for n, p, num_partitions, mp in parameters:
            with self.subTest(f"(n, p, num_partitions, mp) = ({n}, {p}, {num_partitions}, {mp})"):
                adjlist = util.erdos_renyi_adjlist(n, p)

                core, extended = partitioner.partition(None, adjlist, num_partitions, mp)
                
                correct_extension = [set(util.descendants_at_distance_multisource(adjlist, part, distance=mp)).union(part) for part in core]
                
                for a, b in zip(extended, correct_extension):
                    self.assertSetEqual(a, b)
                    
class TestMetisCythonPartitioner(BasePartitionerTest, unittest.TestCase):
    partitioner_class = MetisCythonPartitioner
    
class TestMetisWrapperPartitioner(BasePartitionerTest, unittest.TestCase):
    partitioner_class = MetisPartitioner