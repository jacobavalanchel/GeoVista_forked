
# GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization

<!-- <div align="center">
[💻 Code](https://github.com/ekonwang/GeoVista) | 📃 Paper | [🤗 GeoVista-RL-6k-7B](https://huggingface.co/LibraTree/GeoVista-RL-6k-7B)
</div> -->

<div align="center">
<a href="https://github.com/ekonwang/GeoVista">💻 Code</a> | 📃 Paper | <a href="https://huggingface.co/LibraTree/GeoVista-RL-6k-7B">🤗 GeoVista-RL-6k-7B</a>
</div>


<!-- ![](./assets/visuothink.png) -->
![](./assets/agentic_pipeline.png)

## Quick Start

1. Setup the environment:
```bash
conda create -n geo-vista python==3.10 -y
conda activate geo-vista

bash setup.sh
```

2. Set up web search API key

- We use [Tavily]()



## Benchmarks

- We have already released the [GeoBench](https://huggingface.co/datasets/LibraTree/GeoBench) Dataset on HuggingFace 🤗.

## Citation
Please consider citing our paper and starring this repo if you find them helpful. Thank you!
```bibtex
@misc{wang2025visuothinkempoweringlvlmreasoning,
      title={VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search}, 
      author={Yikun Wang and Siyin Wang and Qinyuan Cheng and Zhaoye Fei and Liang Ding and Qipeng Guo and Dacheng Tao and Xipeng Qiu},
      year={2025},
      eprint={2504.09130},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.09130}, 
}
```
