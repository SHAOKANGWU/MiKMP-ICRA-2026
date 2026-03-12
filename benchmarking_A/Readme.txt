In this folder, we bechmark several algorithms:

1. ProMP_benchmark.py -> the origin ProMP algoroithm
2. KMP_benchmark.py -> the origin KMP algorithms
3. FastKMP.py -> the Fast KMP algorithm that avoid computing a large inverse matrix.

4. Mixture_KMP_benchmark.py -> Using (multi-processer) and (local KMPs) to speed up the computation of KMP
5. MiKMP -> Using (multi-processer) and (local KMPs) to speed up the computation of FastKMP

Run below command for start benchmarking:
    `python MiKMP_benchmark.py'