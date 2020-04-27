# Data Preparation

## ScanNet
Note that our code is largely borrowed from [PointCNN](https://github.com/yangyanli/PointCNN/tree/master/data_conversions). Many thanks here.
<br>

+ Refer to [ScanNet](https://github.com/scannet/scannet#data-organization) to get access to raw ScanNet data. Among all those data, you only need to download  `<scanId>_vh_clean_2.0.010000.segs.json`, `<scanId>_vh_clean_2.ply`, `<scanId>.aggregation.json`. Put them under `/data_extraction/ScanNet/data/`. The data folder's structure would be like.

    ```
    data_extraction
    ├── ScanNet
    |   ├── data
    |   |   ├── <scanId>
    |   |   |   ├── <scanId>_vh_clean_2.0.010000.segs.json
    |   |   |   ├── <scanId>_vh_clean_2.ply
    |   |   |   └── <scanId>.aggregation.json
    |   |   └── ...
    |   ├── benchmark
    |   ├── ...
    ```
    
+ Run `extract_scannet_objs_revised.py`. It  generates `.pts` and `.ply` data:

    ``` python3 extract_scannet_objs_revised.py -f data -b benchmark -o cls_10_pts -s```

+ Run `prepare_scannet_cls_data_new10.py`. It gives out h5 data, which can be directly used in our PointDAN.

     ```python3 prepare_scannet_cls_data_new10.py -f cls_10_pts -o scannet_h5```


    
