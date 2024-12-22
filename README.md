# 202412-12-Dynamic-High-Frequency-Trading-Model

## Overview
This project implements a dynamic high-frequency trading model using GRU (Gated Recurrent Unit) for predictions and leverages Apache Airflow for workflow orchestration. The project outputs include portfolio values, portfolio positions, and an interactive portfolio dashboard.

## Project Structure
```
202412-12-Dynamic-High-Frequency-Trading-Model/
│
├── notebooks/                     
│   ├── project_gru.ipynb          # GRU model generation notebook
│
├── dags/                          
│   ├── project_dag.py             # Airflow DAG script
│
├── output/                        
│   ├── data/                      
│   │   ├── portfolio_values.csv   # Portfolio values data
│   │   ├── portfolio_positions.csv # Portfolio positions data
│   │
│   ├── html/                      
│       ├── portfolio_dashboard.html # Portfolio dashboard visualization
│
├── .gitignore                     
├── README.md                      
└── requirements.txt               
```

## Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
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
- **Portfolio Values**: Stored in `output/data/portfolio_values.csv`.
- **Portfolio Positions**: Stored in `output/data/portfolio_positions.csv`.
- **Dashboard**: Interactive visualization in `output/html/portfolio_dashboard.html`.

## Dependencies
Install all required Python packages with:
```bash
pip install -r requirements.txt
```

## Notes
- Ensure Apache Airflow is installed and properly configured.
- The `portfolio_dashboard.html` file provides a snapshot of portfolio insights.
