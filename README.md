# Smart Irrigation System 

An end-to-end automated smart irrigation framework that leverages machine learning to optimize water consumption. The system evaluates telemetry from 20 environmental sensors to simultaneously predict real-time irrigation requirements across 3 distinct land parcels.

## Key Highlights and Impact
* **High-Accuracy Modeling:** Achieved high precision in localized watering patterns compared to traditional time-based schedules, mitigating water waste.
* **Resource Efficiency:** Designed to dynamically scale water delivery, capable of reducing total agricultural water consumption by over 30%.
* **Multi-Output Architecture:** Implemented an ensemble machine learning approach to predict multiple target zones simultaneously from a single input feature vector.

---

## Technical Stack and Architecture
* **Language:** Python
* **Data Engineering:** Pandas, Numpy, Scikit-Learn (MinMaxScaler)
* **Machine Learning:** RandomForestClassifier wrapped inside a MultiOutputClassifier
* **User Interface:** Streamlit Engine
* **Deployment Readiness:** Serialization using Joblib for model and scaler persistence

### Data and Feature Pipeline
* **Features (X):** 20 continuous variables representing sensor telemetry (`sensor_0` to `sensor_19`) tracking environmental metrics such as soil moisture, ambient temperature, and humidity.
* **Targets (y):** 3 independent binary outputs (`parcel_0`, `parcel_1`, `parcel_2`) where `1` signals active irrigation required and `0` signals standalone soil saturation.

---

## Repository Structure
```text
smart-irrigation/
│
├── data/
│   └── irrigation_machine.csv      # Telemetry and irrigation logs
│
├── Irrigation_System.py            # Primary model training script
├── smart_irrigation_model.py       # Core ML logic and helper functions
├── app.py                          # Streamlit application interface
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
