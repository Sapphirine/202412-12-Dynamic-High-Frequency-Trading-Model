# 202412-12-Dynamic-High-Frequency-Trading-Model

## Overview
This project implements a dynamic high-frequency trading model using GRU (Gated Recurrent Unit) for predictions and leverages Apache Airflow for workflow orchestration. The project outputs include portfolio values, portfolio positions, an interactive portfolio dashboard, and a trained model file.

## Project Structure
```
202412-12-Dynamic-High-Frequency-Trading-Model/
│
├── dags/                          
│   ├── project_dag.py                 # Airflow DAG script
│
├── notebooks/                     
│   ├── project_gru.ipynb              # GRU model generation notebook
│
├── output/                        
│   ├── data/                      
│   │   ├── portfolio_values.csv       # Example portfolio values data
│   │   ├── portfolio_positions.csv    # Example portfolio positions data
│   │
│   ├── html/                      
│   │   ├── portfolio_dashboard.html   # Example portfolio dashboard visualization
│   │
│   ├── model/                    
│       ├── gru.pth                    # Trained GRU model file
│                 
├── README.md                      
└── requirements.txt               
```

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Sapphirine/202412-12-Dynamic-High-Frequency-Trading-Model.git
   cd 202412-12-Dynamic-High-Frequency-Trading-Model
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the project components:
   - Run the Jupyter Notebook: `notebooks/project_gru.ipynb` for GRU model generation.
   - Use the Airflow DAG: `dags/project_dag.py` to automate workflows.

## Outputs
- **Portfolio Values**: Example data stored in `output/data/portfolio_values.csv`.
- **Portfolio Positions**: Example details stored in `output/data/portfolio_positions.csv`.
- **Dashboard**: Example visualization in `output/html/portfolio_dashboard.html`.
- **Trained Model**: The trained GRU model is saved as `output/models/gru.pth`.

## Dependencies
The project requires the following Python libraries (install only these, as others are part of Python's standard library):
- airflow
- yfinance
- pandas
- numpy
- torch
- pandas_market_calendars
- cvxpy
- pendulum
- plotly
- scipy
- sklearn
- matplotlib

To install all dependencies, use:
```bash
pip install -r requirements.txt
```

## Contributors
- **Honghao Huang (hh3042)**
- **Yining Gan (yg2960)**

## Notes
- Ensure Apache Airflow is installed and properly configured.
- The `portfolio_dashboard.html` file provides an example of portfolio insights.
- The `gru.pth` file contains the saved GRU model for reuse.
