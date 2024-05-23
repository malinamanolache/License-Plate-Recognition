## Datasets

|Name|Number of images|Vehicle type|Image resolution|Annotations|Availability|Link|
|:---:|:-----------------:|:----------------:|----|-----------|:----:|:----:|
|**UFPR-ALPR**|4,500|car, motorcycle|1920x1080| - vehicle position and type,<br> - LP position and text <br> - position of characters in LP|By request|[Link](https://arxiv.org/pdf/1802.09567.pdf)
|**RODOSOL-ALPR**|20,000|car, motorcycle, bus, truck|1280x720|- vehicle type <br> - LP layout <br> - text <br> - LP position | By request| [Link](https://github.com/raysonlaroca/rodosol-alpr-dataset/tree/main)|
|**AOLP**|2,049|car|N/A|Not mentioned|By request|[Link](https://github.com/AvLab-CV/AOLP)|
|**CCPD** (Chinese City Parking Dataset, ECCV)|300k|car|720x1160|- LP area and tilt <br> - LP coordinates <br> - LP number <br> - brightness and bluriness of LP|Public|[Link](https://github.com/detectRecog/CCPD)|
|**CalTech Cars**|126|car|896 x 592|not provided but a lot of people use it for testing|Public|[Link](https://data.caltech.edu/records/fmbpr-ezq86)|
|**PKU**|4,000|car|N/A|-LP segmenattions mask|Public|[Link](https://github.com/ofeeler/LPR/tree/master)|

## Papers *i added a comments collumn -> scroll right

|Title|Date|Architecture|Train dataset|Eval datasets|Code|Link|Comments|
|-----|----|------------|-------------|-------------|----|----|--------|
|An Efficient and Layout-Independent Automatic License Plate Recognition System Based on the YOLO detector|March 2021|YOLO-based|Caltech Cars, EnglishLP, UCSD-Stills, ChineseLP, AOLP, SSIG-SegPlate, UFPR-ALPR|As for train|Configuration of the YOLO provided [here](https://web.inf.ufpr.br/vri/publications/layout-independent-alpr/)|[Link](https://arxiv.org/pdf/1909.01754v4.pdf)|none|
|Rethinking and Designing A High-performing Automatic License Plate Recognition Approach|June 2021|CNN|CCPD, AOLP|CCPD, AOLP, PKUData, CLPD|Not found|[Link](https://arxiv.org/pdf/2011.14936.pdf)|none|
|A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector|April 2018|YOLO|UFPR-ALPR,  SSIG|UFPR-ALPR,  SSIG|Official implementation not found|[Link](https://arxiv.org/pdf/1802.09567v6.pdf)|none|
|License Plate Detection and Recognition in Unconstrained Scenarios| 2018 | CNN | subsets from CarsDataset, SSIG, AOLP manually annotated by the authors|OpenALPR, SSIG, AOLP|[Code](http://sergiomsilva.com/pubs/alpr-unconstrained/)|[Link](http://sergiomsilva.com/pubs/alpr-unconstrained/)|none|
|Persian License Plate Recognition System (PLPR)|2023|YOLO|[IR-LPR](https://github.com/mut-deep/IR-LPR), [Iranis-Dataset](https://github.com/alitourani/Iranis-dataset), [ILPR](https://github.com/amirmgh1375/iranian-license-plate-recognition)|Not mentioned, probably same as training|[Code](https://github.com/mtkarimi/persian-license-plate-recognition)|No paper|none|
|LicensePlateDetector|Last commit 8 months ago|NeuralNetwork for recognizing characters, Connected Component Analysis|Personal dataset|Not mentioned|[Code](https://github.com/apoorva-dave/LicensePlateDetector)|No paper|none|
|Open-LPR|2023|-|-|-|[Code](https://github.com/faisalthaheem/open-lpr?tab=readme-ov-file)|No paper|none|
|A Real-Time License Plate Detection Method Using a Deep Learning Approach|2021|YOLOv3|-|-|[Code](https://github.com/alitourani/yolo-license-plate-detection) (doesn't have OCR)|It has paper but need to enter with university account |none|
|A Flexible Approach for Automatic License Plate Recognition in Unconstrained Scenarios|2021|YOLO|-still figuring it out-|UFPR-ALPR, OpenALPR-BR, AOLP, CD-HARD, ?CCPD?|(it needs car cropped first. Other repos might need that too)(It also needs you to specify the type of vehicle: car, bus, truck)They used this: https://github.com/claudiojung/iwpod-net|[Link](https://sci-hub.yncjkj.com/10.1109/tits.2021.3055946)|Localization works, but it doesn't include OCR. An ideea is to use Sergio's OCR from "License Plate Detection and Recognition in Unconstrained Scenarios"|
|Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers|--|--|RODOSOL|Not mentioned|[Code](https://github.com/valfride/lpr-rsr-ext/)|[Link](https://www.sciencedirect.com/science/article/pii/S0097849323000602?casa_token=C_6dj-w5wyAAAAAA:vzMhqD4sx7cB-b2oquDx3NZroJVgPIuP5MQUpl6Ix_i8z8hBi1b2QCi7T_53dfLhkjxe2Lf4RtI)|Just OCR - tested on RodoSol, also has git implementation.|
