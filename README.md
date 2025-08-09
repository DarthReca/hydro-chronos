<div align="center">
  
## HydroChronos: Forecasting Decades of Surface Water Change

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Eleonora Poeta**](https://github.com/eleonorapoeta)<sup>1</sup> · [**Eliana Pastor**](https://elianap.github.io/)<sup>1</sup> · [**Isaac Corley**](https://isaacc.dev/)<sup>2</sup>

[**Tania Cerquitelli**](https://smartdata.polito.it/members/tania-cerquitelli)<sup>1</sup> · [**Elena Baralis**](https://smartdata.polito.it/members/elena-baralis/)<sup>1</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy&emsp;<sup>2</sup>Wherobots, USA

[SIGSPATIAL 2025](https://sigspatial2025.sigspatial.org/)

<a href="https://arxiv.org/abs/2506.14362"><img src='https://img.shields.io/badge/Paper-%23B31B1B?style=flat&logo=arxiv&logoColor=%23B31B1B&labelColor=white' alt='Paper PDF'></a>
<a href="https://huggingface.co/datasets/DarthReca/hydro-cronos"><img src='https://img.shields.io/badge/HydroChronos-yellow?style=flat&logo=huggingface&logoColor=yellow&labelColor=grey'></a>
<a href="https://huggingface.co/collections/DarthReca/actu-6872e67bacfcdbef020e25ff"><img src='https://img.shields.io/badge/ACTU-yellow?style=flat&logo=huggingface&logoColor=yellow&labelColor=grey'></a>

</div>


**In this paper, we introduce HYDROCHRONOS, a large-scale, multi-modal spatiotemporal dataset designed for forecasting surface water dynamics. The dataset provides over three decades of aligned Landsat 5 and Sentinel-2 imagery, coupled with climate data and Digital Elevation Models for lakes and rivers across Europe, North America, and South America. We also propose AquaClimaTempo UNet, a novel spatiotemporal architecture with a dedicated climate data branch.** Our findings show that our model significantly outperforms a Persistence baseline in forecasting future water dynamics by +14% and +11% F1-scores across change detection and direction of change classification tasks, respectively, and by +0.1 MAE on the magnitude of change regression. Additionally, we conduct an Explainable AI analysis to identify the key variables and input channels that influence surface water change, offering insights to guide future research.

## Getting Started

## Dataset
The dataset is available on [HuggingFace](https://huggingface.co/datasets/DarthReca/hydro-cronos).

### Data Modalities
The dataset comprises Landsat-5 (L) TOA and Sentinel-2 (S) TOA images. There are 6 coherently aligned bands for both satellites:

| Landsat | Sentinel | Description | Central Wavelength (L/S) |
|:-------:|:--------:|:-----------:|:------------------------:|
|    B1   |    B2    |     Blue    |        485/492 nm        |
|    B2   |    B3    |    Green    |        560/560 nm        |
|    B3   |    B4    |     Red     |        660/665 nm        |
|    B4   |    B8    |     NIR     |        830/833 nm        |
|    B5   |    B11   |     SWIR    |       1650/1610 nm       |
|    B7   |    B12   |     SWIR    |       2220/2190 nm       |

They are coupled with climate variables from [TERRACLIMATE](https://www.nature.com/articles/sdata2017191) and Copernicus GLO30-DEM.

## Models

You can easily load the model with HuggingFace. Each repository contains different configurations of ACTU.

| Task | Weights |
| :--- | :--- |
| **Change Detection** | [Link](https://huggingface.co/DarthReca/actu-change-detection) |
| **Direction Classification** | [Link](https://huggingface.co/DarthReca/actu-direction-classification) |
| **Magnitude Regression** | [Link](https://huggingface.co/DarthReca/actu-magnitude-regression) |

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@misc{cambrin2025hydrochronosforecastingdecadessurface,
      title={HydroChronos: Forecasting Decades of Surface Water Change}, 
      author={Daniele Rege Cambrin and Eleonora Poeta and Eliana Pastor and Isaac Corley and Tania Cerquitelli and Elena Baralis and Paolo Garza},
      year={2025},
      eprint={2506.14362},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.14362}, 
}
```
