# 🗺️ Abuja Route Speed Predictor with GIS Visualization

An intelligent transportation analytics platform built with **Python, Streamlit, Machine Learning, and GIS Visualization** to predict average vehicle speeds, analyze transportation routes, and visualize Abuja's road network through interactive maps.

This project demonstrates how **Artificial Intelligence, Data Science, and Geospatial Analytics** can be integrated into an intelligent transportation system for smarter mobility and data-driven urban planning.

---

# 🌐 Live Application

🔗 https://kola-abuja-route.streamlit.app/

---

# 📌 Project Overview

Traffic congestion continues to impact travel efficiency, productivity, and transportation planning in rapidly growing cities such as Abuja.

This application was developed to demonstrate how Machine Learning and GIS technology can be combined to analyze transportation routes, predict vehicle speeds, and provide interactive visualization for better transportation decision-making.

The platform allows users to:

- Upload custom transportation datasets
- Predict average vehicle speed using Machine Learning
- Visualize transportation routes on an interactive GIS map
- Evaluate model performance
- Compare predicted and actual speeds
- Export prediction results

---

# 🎯 Problem → Solution → Impact

## Problem

Transportation planners often require tools capable of analyzing traffic performance while visualizing transportation networks in an intuitive manner.

## Solution

This application combines **Machine Learning, GIS visualization, interactive dashboards, and transportation analytics** into a single platform that predicts average travel speed and displays road networks geographically.

## Impact

The project demonstrates how AI can support:

- Traffic speed prediction
- Smart mobility planning
- GIS transportation analysis
- Transportation decision support
- Smart city development

---

# 🚀 Key Features

## 📁 Flexible Dataset Upload

Users can upload:

- CSV files
- Excel files

or use the built-in **Maitama sample dataset** for demonstration.

---

## 🤖 Machine Learning Prediction

The application uses **Linear Regression** to estimate average vehicle speed using:

- Route Length (km)
- Travel Time (seconds)

---

## 📊 Model Evaluation

Displays important performance metrics including:

- R² Score
- Mean Absolute Error (MAE)

---

## 🗺️ Interactive GIS Route Map

Built with **PyDeck**, the GIS dashboard displays:

- Road connections
- Speed-based route colouring
- Start and destination points
- Interactive hover information

---

## 📥 Export Prediction Results

Users can download prediction results for further analysis in CSV format.

---

# 📸 Application Screenshots

## 🖥️ Dashboard

![Dashboard](assets/Dashboard.png)

---

## 🔮 Speed Prediction

![Prediction](assets/Predict.png)

---

## 📊 Application Summary

![Application Summary](assets/App%20Summary.png)

---

# 🧠 Machine Learning Model

## Algorithm

**Linear Regression**

### Input Features

- Route Length (km)
- Travel Time (seconds)

### Predicted Output

- Average Vehicle Speed (km/h)

---

# 📊 Model Development & Evaluation

This project was developed as a prototype to demonstrate the complete workflow of a Machine Learning-powered transportation analytics system.

The application combines:

- Data preprocessing
- Machine Learning prediction
- GIS visualization
- Interactive dashboards
- Data export functionality

into a single intelligent platform.

During development, one important observation was that model evaluation metrics varied slightly between application runs.

This occurred because different training and testing samples were generated during model evaluation. With relatively small datasets, slight variations in train-test splits naturally produce different evaluation results.

This reinforced an important lesson in Machine Learning:

> **Reliable model performance depends not only on the algorithm selected but also on sufficient training data, feature quality, and reproducible evaluation strategies.**

Future versions of this application will incorporate larger transportation datasets, enhanced feature engineering, additional regression algorithms, and more robust model evaluation techniques.

---

# 🏗️ System Architecture

```text
Transportation Dataset
         │
         ▼
Data Upload & Validation
         │
         ▼
Data Preprocessing (Pandas)
         │
         ▼
Machine Learning Model
(Linear Regression)
         │
         ├───────────────┐
         ▼               ▼
Speed Prediction    Model Evaluation
         │               │
         └───────┬───────┘
                 ▼
GIS Visualization (PyDeck)
                 ▼
Interactive Streamlit Dashboard
                 ▼
Prediction Export (CSV)
                 ▼
Transportation Decision Support
```

---

# 🛠️ Technology Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- Linear Regression

## Data Analysis

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Geospatial Analytics

- PyDeck

## Web Framework

- Streamlit

---

# 📂 Project Structure

```text
Abuja-Route-Speed-Predictor-with-GIS-Visualization/
│── assets/
│   ├── Dashboard.png
│   ├── Predict.png
│   ├── App Summary.png
│── Rout.py
│── requirements.txt
│── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/kola56de/Abuja-Route-Speed-Predictor-with-GIS-Visualization.git

cd Abuja-Route-Speed-Predictor-with-GIS-Visualization
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run Rout.py
```

---

# 🎯 Applications

- Intelligent Transportation Systems
- Traffic Speed Prediction
- Smart Mobility
- GIS Transportation Analytics
- Urban Transport Planning
- Transportation Decision Support
- Smart City Development

---

# 📈 Future Roadmap

- Google Maps Traffic API Integration
- Real-Time Traffic Monitoring
- Congestion Hotspot Detection
- Route Recommendation Engine
- Public Transport Optimization
- Power BI Executive Dashboard
- Mobile Application
- Deep Learning Traffic Prediction
- Multi-District Expansion

---

# 👨‍💻 Author

## **Engr. Dr. Kolade Olonisakin, FNSE**

**Civil Engineer | Data Scientist | Machine Learning Engineer | AI Engineer | Transportation & GIS Analytics**

🌍 **Portfolio**

https://olonisakin-emmanuel.github.io/OlonisakinEmmanuel.github.io/

💼 **LinkedIn**

https://www.linkedin.com/in/engr-dr-kolade-olonisakin-fnse/

💻 **GitHub**

https://github.com/kola56de

---

# ⭐ Support

If you found this project useful, please consider giving it a **⭐ Star** on GitHub.

Feedback, suggestions, and collaboration opportunities are always welcome.
