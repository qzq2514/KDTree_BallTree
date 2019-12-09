# KDTree_BallTree
Python implement of KD_Tree and Ball_Tree


This is python implement of KD_Tree and Ball_Tree based on [winequality-white.csv](http://archive.ics.uci.edu/ml/datasets/Wine+Quality),
This dataset has 4898 samples from 10 quality level.Each sample has 12 attributes.In this program,80% samples are spilted to training data and the left are used to test.  

# Usage:  
```python BallTree_KNN.py```  


```python KDTree_KNN.py```

# Result:

| Type | Construction time(s) | time(s)/search |compare times/search|MSE|
| - | :-: |:-: |:-: | -: |
| KD Tree（K=5） | 0.021|0.089| 1256.76| 0.842 |
| Ball Tree (K=5)| 0.931|0.063| 869.01 | 0.815 |



