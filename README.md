# ABC

**ABC** is a unified framework designed for handling long-tail and out-of-distribution (OOD) tasks, without the need for auxiliary OOD data.

---

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Environment Setup](#environment-setup)
3. [Training](#training)
4. [Testing](#testing)
5. [Acknowledgments](#acknowledgments)

---

## Data Preparation

Ensure that your data is organized according to the configurations specified in the `configs/` directory. The structure and paths should match the `data_loader` settings in the corresponding configuration files.

### **ID Datasets**

For preparing datasets such as CIFAR-10, CIFAR-100, and ImageNet, please follow the instructions provided [here](https://github.com/Vanint/SADE-AgnosticLT/tree/main).

### **OOD Datasets**
The OOD datasets for testing can be obtained through the links below:

- **SVHN**: [link](http://ufldl.stanford.edu/housenumbers/)
- **TinyImageNet**: [link](http://image-net.org/download)
- **DTD (Texture)**: [link](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- **places365**: [link](http://places2.csail.mit.edu/)
- **LSUNCrop** & **LSUNResize**: [link](http://lsun.cs.princeton.edu/)
- **iNaturalist2018**: [link](https://www.inaturalist.org/pages/help#data)

### **Directory Structure**

Organize your files as follows:

```plaintext
LTOOD/
├── CIFAR-10/
├── CIFAR-100/
├── data_txt/
├── Imagenet/
├── iNaturalist2018/
└── OOD/
    ├── DTD/
    ├── LSUN/
    ├── LSUNCrop/
    ├── LSUNResize/
    ├── places365/
    ├── SCOOD/
    ├── SVHN/
    └── TinyImageNet/
```

---

## Environment Setup

Install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

---

## Training

1. **Configuration**:
   - Open the configuration file located at `configs/<ID_data>/ir<imbalance_rate>.json`.
   - Update the `data_dir` parameter in `"data_loader": "args"` to point to your dataset directory.

2. **Start Training**:
   Execute the following command to start training:

   ```bash
   python train.py \
       --config <path/to/config>
   ```

   Replace `<path/to/config>` with the path to your configuration file.

---

## Testing

1. **Create a Shell Script**:
   Save the following script as `test.sh`:

   ```bash
   ood_datasets=("TinyImageNet" "LSUNCrop" "LSUNResize" "SVHN" "DTD" "places365")
   device=6
   resume=<path/to/your/checkpoint>

   python test.py \
       -d ${device} \
       -id ID \
       --resume ${resume}

   python test_ood.py \
       -d ${device} \
       -id OOD_iNature \
       --resume ${resume}

   for ood_dataset in ${ood_datasets[@]}; do
       python test_ood.py \
           --ood_data ${ood_dataset} \
           --ood_data_dir ./data/LTOOD/OOD/${ood_dataset} \
           -d ${device} \
           -id OOD_${ood_dataset} \
           --resume ${resume}
   done
   ```

2. **Run the Script**:
   Execute the testing process with:

   ```bash
   bash test.sh
   ```

---

## Notes

- Replace `<path/to/config>` and `<path/to/your/checkpoint>` with the actual paths relevant to your setup.
- The `device` variable refers to the GPU device index. Adjust it based on your hardware configuration.
- Ensure all datasets are located in the appropriate directories as specified in the configuration files.

---

## Acknowledgments

This codebase is adapted from [LGLA](https://github.com/Tao0-0/LGLA).
