## Datasets

|Name|Number of images|Vehicle type|Image resolution|Annotations|Availability|Link|
|:---:|:-----------------:|:----------------:|----|-----------|:----:|:----:|
|**UFPR-ALPR**|4,500|car, motorcycle|1920x1080| - vehicle position and type,<br> - LP position and text <br> - position of characters in LP|By request|[Link](https://arxiv.org/pdf/1802.09567.pdf)
|**RODOSOL-ALPR**|20,000|car, motorcycle, bus, truck|1280x720|- vehicle type <br> - LP layout <br> - text <br> - LP position | By request| [Link](https://github.com/raysonlaroca/rodosol-alpr-dataset/tree/main)|
|**AOLP**|2,049|car|N/A|Not mentioned|By request|[Link](https://github.com/AvLab-CV/AOLP)|
|**CCPD** (Chinese City Parking Dataset, ECCV)|300k|car|720x1160|- LP area and tilt <br> - LP coordinates <br> - LP number <br> - brightness and bluriness of LP|Public|[Link](https://github.com/detectRecog/CCPD)|
|**CalTech Cars**|126|car|896 x 592|not provided but a lot of people use it for testing|Public|[Link](https://data.caltech.edu/records/fmbpr-ezq86)|
|**PKU**|4,000|car|N/A|-LP segmenattions mask|Public|[Link](https://github.com/ofeeler/LPR/tree/master)|

## Papers

|Title|Date|Architecture|Train dataset|Eval datasets|Code|Link|
|-----|----|------------|-------------|-------------|----|----|
|An Efficient and Layout-Independent Automatic License Plate Recognition System Based on the YOLO detector|March 2021|YOLO-based|Caltech Cars, EnglishLP, UCSD-Stills, ChineseLP, AOLP, SSIG-SegPlate, UFPR-ALPR|As for train|Configuration of the YOLO provided [here](https://web.inf.ufpr.br/vri/publications/layout-independent-alpr/)|[Link](https://arxiv.org/pdf/1909.01754v4.pdf)|
|Rethinking and Designing A High-performing Automatic License Plate Recognition Approach|June 2021|CNN|CCPD, AOLP|CCPD, AOLP, PKUData, CLPD|Not found|[Link](https://arxiv.org/pdf/2011.14936.pdf)|
|A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector|April 2018|YOLO|UFPR-ALPR,  SSIG|UFPR-ALPR,  SSIG|Official implementation not found|[Link](https://arxiv.org/pdf/1802.09567v6.pdf)|
