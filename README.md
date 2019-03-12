# TargetSpecificDomainAdaptation
This is the implementation of the paper "Cross-Domain Sentiment Classification with Target Domain Specific Information" published in ACL 2018.

## Running Example
For peroformance of our proposed method on task "0->1", run:\\
CUDA_VISIBLE_DEVICES=0 python3 co_training.py 0 1
\\
Here, "0, 1, 2, 3" denotes the "book", "dvd", "electronic" and "kitchen" domain, resprectively.

