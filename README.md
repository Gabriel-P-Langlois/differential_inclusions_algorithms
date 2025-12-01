# QUADPROG project

This is the start of the repository for the second differential inclusions
paper.

TODO:

1) Update the nnls_hinge_qr, if necessary.
2) Change name to qp_incl.m and write qp_incl_qr.m file. This one
   uses the QR decomposition to update the problem. Verify correctness
   and benchmark in a third test vs other algorithms.

1) Benchmark all the nonnegative least-squares (NNLS) algorithms. These
   are the method of hinges (vanilla; with QR decomposition), MATLAB's
   standard lsqnonneg method, projected gradient descent, the asymptotic 
   method \w differential inclusions v1 and v2 with linsolve at the end. 
   Maybe Method of Hinges \w QR method + asymptotic method as well?

2) Once all the methods have been benchmarked, write the QUADPROG version
   with the Method of Hinges + QR decomposition. Compare with differential
   inclusions alone; consider combining them (differential inclusions until
   the last solve? This way it preserves the accuracy...). Maybe just do
   the first point and optimize later. Get code that works correct and
   beats MATLAB's linprog and quadprog to a pulp.

3) Once that is ready, update the manuscript; go over the TODO list and
   update the manuscript to the point where the draft stands on its own
   as an arxiv preprint (but do not submit).
   