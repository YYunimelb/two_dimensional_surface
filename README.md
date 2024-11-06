# 2D Layered Materials Analysis Project

This project provides a computational framework for analyzing and characterizing two-dimensional (2D) layered materials, with a focus on identifying structures that exhibit layered characteristics at the atomic level. It includes tools for structural transformations, layer detection, and material standardization to aid in the study of materials with potential applications in areas such as nanotechnology, electronics, and spintronics.

## Overview

The 2D Layered Materials Analysis Project is designed to streamline the process of identifying and analyzing layered structures within a variety of materials. The project includes modules for:
- Parsing and processing structure files (e.g., VASP POSCAR files)
- Building supercells and calculating interatomic distances
- Determining layered characteristics based on atomic arrangements
- Standardizing material structures into normalized coordinate systems

## Features

1. **Structure Parsing and Processing**: Extract atomic coordinates, lattice vectors, and atomic types from structure files. This allows for efficient analysis of material properties.

2. **Supercell Creation**: Generate supercells based on specified replication boundaries to better observe and analyze interlayer interactions.

3. **Layer Identification**: Identify if a material has a layered structure by checking for unique interatomic distances and atomic plane configurations.

4. **Coordinate Standardization**: Transform atomic positions to normalized bases and relative coordinates, providing a standardized structure file for further analysis.

5. **Data Matching**: The project also supports integration with the Materials Project database for comparison and matching of experimental structures with known materials.

## Project Structure

- `parsers/`: Structure file parsing tools, including VASP POSCAR parser.
- `utils/`: Utility functions for operations such as supercell generation, distance calculations, and geometry transformations.
- `models/`: Data models that represent atomic structures, lattice parameters, and associated attributes.
- `processors/`: Classes for handling various stages of structure processing, including supercell creation and layer checking.
- `normalizers/`: Functions and classes to convert material structures to a standard format and to ensure consistent coordinate systems.


