![image](https://github.com/user-attachments/assets/c67479bb-ca81-440f-a252-38e5361944e1)
## SDDT-SR

The pytorch implementation for **SDDT-SR** in paper "[SDDT-SR: A Scale-Decoupling Super-Resolution Network With Domain Transfer for Heterogeneous Images](https://ieeexplore.ieee.org/document/10769009)" 
on [IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4609443).  

## Requirements
- Python 3.6
- Pytorch 1.7.0


## Datasets
### Sentinel 2-Google Image Super-Resolution Dataset (SGSRD)
A dataset dedicated to image super-resolution in heterogeneous remote sensing images. Sentinel-2 images and Google Images were chosen as sources, as a way to fit the requirement that LRI and HRI come from different sensors and have large scale on resolution differences. We downloaded sentine1-2 images and Google satellite images from Zengcheng District, Guangzhou City, Guangdong Province, China, from 2018 to 2020. The sentinel-2 images with 10-metre resolution are used as the LRIs, while the 1.3-m resolution Google images are used as the HRIs. By nonoverlapping sampling, the sentinel-2 images were cropped into patches of size of 64 × 64, and the Google images were cropped into patches of size of 512 × 512 accordingly

- Download the SGSRD Dataset: [Baidu](https://pan.baidu.com/s/19GUCqkx8qJ9o6wA4o6Qc-A?pwd=9474)

## Citation

Please cite our paper if you use this code in your work:

```
@ARTICLE{10769009,
  author={Lin, Simin and Ru, Jiangdi and Li, Jiaqi and Wen, Keyao and Liu, Mengxi and Shi, Qian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={SDDT-SR: A Scale-Decoupling Super-Resolution Network With Domain Transfer for Heterogeneous Images}, 
  year={2025},
  volume={18},
  number={},
  pages={2052-2062},
  keywords={Image reconstruction;Human-robot interaction;Superresolution;Remote sensing;Generative adversarial networks;Accuracy;Generators;Training;Spatial resolution;Frequency-domain analysis;Deep learning (DL);heterogeneous images;remote sensing;super-resolution;unpaired dataset},
  doi={10.1109/JSTARS.2024.3507104}}
```
