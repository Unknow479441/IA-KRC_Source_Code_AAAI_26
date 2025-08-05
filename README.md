# Project Description

Complete code and models have been uploaded to https://github.com/Unknow479441/IA-KRC_Source_Code_AAAI_26.git

## 1. Environment Requirements

### System Requirements

```
Operating System:         Windows 10
CPU:                     AMD EPYC 9554
GPU:                     RTX 4090 x2
RAM:                     DDR5 4800MHz 192GB
```

### Core Training Dependencies

```
python                    3.9.21
torch                     2.0.1+cu118
cudatoolkit               11.3.1
numpy                     1.26.0
sacred                    0.8.7
pysc2                     4.0.0
s2clientprotocol          5.0.14.93333.0
s2protocol                5.0.14.93333.0
pyyaml                    6.0.2
```

## 2. Installation

### StarCraft II Environment Setup

To install the StarCraft II environment, run the following command:

```bash
install_sc2.bat
```

This script will automatically download and configure the StarCraft II environment required for training.

### Map Configuration

After StarCraft II installation, copy the custom map file to override the default map:

```bash
# Copy the custom map file
copy "src\envs\starcraft2\maps\SMAC_Maps\empty_passive.SC2Map" "StarCraft II\Maps\SMAC_Maps\"
```

This will replace the default map with the custom empty_passive map required for training.

## 3. Training

### Start Training

To start training with Dense-obstacle map 12v12, IA-KRC_VS_Vision configuration:

```bash
cd src
python main.py
```

This will launch the training process using the configured environment and algorithms.

