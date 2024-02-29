
# LineageVAE

## Abstract
LineageVAE is a groundbreaking computational tool designed to map the gene expression journey of cells over time, leveraging single-cell RNA sequencing data. Unlike traditional methods that cannot track changes due to the destructive nature of sequencing, LineageVAE employs deep learning to trace cell lineage, offering insights into cell development processes. It effectively predicts the transitions and regulatory dynamics of cells back to their progenitor states with high precision, making it invaluable for studies in developmental biology and beyond.

This project is a collaborative effort led by the Teppei Shimamura Lab ([Tokyo Medical and Dental University, Tokyo and Nagoya University, Nagoya](https://www.shimamlab.info/)) and the Yasuhiro Kojima Lab ([National Cancer Center Research Institute, Tokyo](https://www.ncc.go.jp/jp/ri/division/computational_life_science/index.html)), and was developed by Koichiro Majima.

## Installation

To install LineageVAE, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/LzrRacer/LineageVAE.git
```

2. Navigate to the LineageVAE directory:

```bash
cd LineageVAE
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use LineageVAE, follow these instructions:

1. Prepare your single-cell RNA sequencing dataset according to the guidelines provided in the `data` directory.

2. Run LineageVAE to analyze your dataset and trace cell lineage:

```bash
python lineagevae.py --input your_dataset.csv --output output_directory
```

Replace `your_dataset.csv` with the path to your dataset file and `output_directory` with the path where you want the results to be saved.

For more detailed usage instructions and advanced options, refer to the documentation in the `docs` folder.

## Support

For questions and support, please open an issue in the [GitHub repository](https://github.com/LzrRacer/LineageVAE/issues).

## Contributing

Contributions to LineageVAE are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.