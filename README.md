# [TMM'25] STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario

This repository contains the official PyTorch implementation of our paper:  
[**"STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario"**](https://ieeexplore.ieee.org/abstract/document/10855612/).  

---

## Installation

Please make sure you have [conda](https://docs.conda.io/en/latest/) installed.  
Then create and activate the environment as follows:

```bash
conda env create -f environment.yml
conda activate stba    # Replace "stba" with your environment name if different
```

## Download Data

You can download the test ImageNet dataset from [Google Drive](https://drive.google.com/drive/folders/11-pBzbZN81dCIVulYsQEWY4RiScy7g7N?usp=sharing).  
Afterwards, move the downloaded files into the `data` folder of this repository.

---

## Running the Attack

To run the STBA frequency-based attack on ImageNet, use:

```bash
CUDA_VISIBLE_DEVICES=1 python -u STBA_Freq.py -D ImageNet
```

- Modify the device ID (`CUDA_VISIBLE_DEVICES=1`) as needed.
- You can change the dataset and other parameters as required.

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## Citation

If you find this work or code useful for your research, please cite our paper:

```bibtex
@article{liu2025stba,
  title={STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario},
  author={Liu, Renyang and Lam, Kwok-Yan and Zhou, Wei and Wu, Sixing and Zhao, Jun and Hu, Dongting and Gong, Mingming},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```

Thank you for your interest and support!
