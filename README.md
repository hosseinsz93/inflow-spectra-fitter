# Spectral Parameter Extraction for Synthetic Turbulence Generator

This is an add-on tool for the [`GenerateInflow.c`](https://github.com/hosseinsz93/GenerateInflow-VFS3.1.git) synthetic turbulence generator. It fits experimental spectral data to the von Kármán spectrum and extracts the parameters needed by the turbulence generation code.

## Purpose

[`GenerateInflow.c`](https://github.com/hosseinsz93/GenerateInflow-VFS3.1.git) requires turbulence parameters (`D_coef`, `ustar`, `L_coef`, `Y_loc`) that are often difficult to determine directly. This tool:
1. Fits experimental frequency spectra to the von Kármán energy spectrum
2. Extracts fundamental turbulence parameters (α, ε, L)
3. Converts them to the format required by [`GenerateInflow.c`](https://github.com/hosseinsz93/GenerateInflow-VFS3.1.git)

## Components

- **spectra.ipynb** - Python notebook for fitting experimental spectra
- **Spectral data.xlsx** - Example experimental spectral data

## Usage

### 1. Prepare Your Data
Format your experimental spectral data in Excel with columns:
- `frequency` [Hz]
- `spectral density` [m²/s]

### 2. Run the Notebook
```bash
jupyter notebook spectra.ipynb
```

The notebook fits the von Kármán spectrum:
$$E(k) = \alpha \varepsilon^{2/3} L^{5/3} \frac{L^4 k^4}{(1 + L^2 k^2)^{17/6}}$$

And extracts:
- **α** - dimensionless amplitude constant
- **ε** - turbulent dissipation rate [m²/s³]
- **L** - integral length scale [m]

### 3. Get GenerateInflow.c Parameters
The notebook outputs the required parameters:
```
-D_coef 2.485
-ustar 0.104
-ustar4mean 0.104
-L_coef 5.078
-Y_loc 0.110
```

Copy these into your `GenerateInflow.inp` configuration file.

## Parameter Conversion

The conversion formulas are:
- `D_coef` = α (directly from spectrum fit)
- `ustar` = (ε × Y_loc)^(1/3) / V_ref (friction velocity)
- `L_coef` = L / Y_loc (normalized length scale)

## Dependencies

```bash
pip install pandas matplotlib numpy scipy openpyxl
```
