# Dynamic Data Layout Optimization with Worst-case Guarantees 
Authors: Kexin Rong, Paul Liu, Sarah Ashok Sonje, Moses Charikar

Paper link: https://arxiv.org/abs/2405.04984

## Environment setup
Run ```pip3 install -r requirements.txt``` to setup the required Python3 dependencies.

## Configuration 
- Data: ```resources/config/*.json``` 
- Query: ```resources/config/*.p``` pickle files with predicates for each query.   
- Overall: ```resources/params/*.json``` 

## Methods of comparison 
To run methods in simulation: 
- Generate candidate data layouts according to sliding window or reservoir sampling of past queries:
```python layout_main.py --config config_name```
- Static baseline: best static layout in hindsight 
```python offline_main.py --config config_name```
- Periodic baseline: switch to new layout with better performance 
```python periodic_main.py --config config_name```
- Regret baseline: switch layouts when cumulative regret > movement cost  
```python regret_main.py --config config_name```
- Modified RANDOM algorithm
```python random_main.py --config config_name```

To measure end-to-end time: 
```
# --alg argument takes one of [offline, random, periodic, regret]
python replay_main.py --config config_name --rewrite --root /path/to/partition --alg offline
```
