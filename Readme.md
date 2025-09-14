# Fantasy Cricket Team Builder

A complete Fantasy Cricket prediction and team optimization project inspired by Dream11. This project predicts player fantasy points, evaluates them using machine learning, and constructs an optimized team under official constraints.

---

## **Features**

- **Player Points Prediction**  
  Predicts fantasy points for players based on historical performance using machine learning models such as RandomForest or XGBoost.

- **Team Optimization**  
  Automatically selects a Dream11-style team using Mixed Integer Linear Programming (PuLP), adhering to constraints like player roles, team limits, and total credits.

- **Interactive Dashboard**  
  Provides an interactive Streamlit dashboard to explore predicted points, simulate teams, and visualize team selection.

---

## **Project Structure**

- `data_players_sample.csv` – Sample dataset of players and stats  
- `data_pipeline.py` – Load, clean, and preprocess data  
- `modeling.py` – Train ML models and make predictions  
- `optimizer.py` – Team selection algorithm using PuLP  
- `app_streamlit.py` – Streamlit dashboard for interactive use  
- `requirements.txt` – Python dependencies  
- `README.md` – Project documentation  

---

## **Installation**

1. Clone the repository.  
2. Create and activate a virtual environment.  
3. Install project dependencies from `requirements.txt`.  

---

## **Usage**

- **Run Predictions** – Train models and predict fantasy points for players using the sample dataset.  
- **Optimize Team** – Generate an optimized Dream11 team within constraints for player roles, team selection, and total credits.  
- **Launch Dashboard** – Use the Streamlit dashboard to explore predictions and team optimization interactively.

---

## **Key Dependencies**

- `pandas` – Data handling  
- `scikit-learn` – Machine learning models  
- `xgboost` – Advanced regression models  
- `PuLP` – Optimization for team selection  
- `streamlit` – Interactive dashboard  

---

## **Future Enhancements**

- Integrate live player stats and match data for real-time predictions  
- Ensemble models for improved prediction accuracy  
- Support for multiple fantasy league formats  
- Advanced visualizations using Plotly or Altair  

---

## **License**

This project is open-source and available under the MIT License.
