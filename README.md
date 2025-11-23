# MOSFET Power Loss & Thermal Modeling using Artificial Neural Networks (ANNs)

## **Overview**
This project utilizes **Artificial Neural Networks (ANNs)** to predict **MOSFET switching losses, conduction losses, and thermal impedance**. The trained models enable **accurate power loss estimation** and **thermal behavior analysis**, crucial for optimizing power electronics applications. The project includes a **web-based interface** built with Flask for interactive predictions and visualization.

## **Key Features**
- **Switching Loss Prediction**: Estimates **Turn-On & Turn-Off energy losses** under different **Gate Resistor (Rgon, Rgoff) values**.  
- **Conduction Loss Model**: Predicts **Drain-Source Voltage (Vds)** based on **Drain Current (Ids) and Temperature** for both MOSFET and Body Diode modes.  
- **Thermal Impedance Model**: Estimates **Junction-to-Case Thermal Impedance (Zth_jc)** over time.  
- **Machine Learning Models**: Trains and saves **ANN-based models** using **PyTorch**.  
- **Web-Based Interface**: Interactive Flask web application with **3D visualizations** and **real-time predictions**.  
- **Multiple Model Support**: Supports multiple MOSFET models with automatic model selection.  
- **Total Loss Calculation**: Combines switching and conduction losses for comprehensive power analysis.

---

## **Project Objectives**
- **Develop ANN models** for **MOSFET switching, conduction, and thermal performance**.
- **Train ANNs using real or simulated datasets** for high-accuracy predictions.
- **Optimize power loss estimation** based on **operating conditions**.
- **Provide a user-friendly web interface** to input test cases and obtain predictions with visualizations.

---

## **Project Structure**

```
Transistor-MOSFETLibraryANN/
│
├── app.py                          # Flask web application (main entry point)
├── C2MSeries_ANN_full_model.py     # ANN model training script
├── C2Mseries_Full_Model_Test.py    # Model testing and validation script
├── requirements.txt                # Python dependencies
├── LICENSE                         # License file
│
├── models/                         # Trained models and data
│   ├── c2m0080120d_ANN_Trainned_Model.pth
│   ├── c2m0080120d_scaler_switching.pkl
│   ├── c2m0080120d_scaler_conduction_mosfet.pkl
│   ├── c2m0080120d_scaler_conduction_diode.pkl
│   ├── c2m0080120d_thermal_data.txt
│   └── C2M0080120D.json
│
├── templates/                      # HTML templates
│   └── index.html                 # Main web interface
│
├── static/                         # Static assets
│   └── models/                    # 3D model files (GLTF/GLB)
│       ├── TO247-3.bin
│       ├── TO247-3.glb
│       └── TO247-3.gltf
│
└── README.md                       # This file
```

---

## **Installation**

### **Prerequisites**
- Python 3.7 or higher
- pip (Python package manager)

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/Transistor-MOSFETLibraryANN.git
cd Transistor-MOSFETLibraryANN
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Dependencies**
- **Flask** (2.3.2) - Web framework
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computations
- **scikit-learn** - Data preprocessing and scaling
- **joblib** - Model serialization
- **pandas** - Data manipulation (for training)
- **matplotlib** - Plotting (for training)

---

## **Usage**

### **Running the Web Application**

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Select a MOSFET model** from the dropdown menu (if multiple models are available).

4. **Choose a calculation type:**
   - **Switching Loss**: Calculate turn-on and turn-off energy losses
   - **Conduction Loss (MOSFET)**: Predict Vds and conduction losses for MOSFET mode
   - **Conduction Loss (Body Diode)**: Predict Vds and conduction losses for body diode mode
   - **Thermal Impedance**: Estimate junction-to-case thermal impedance
   - **Total Loss**: Combined switching and conduction losses

5. **Enter the required parameters** and click **Calculate**.

### **Input Parameters**

#### **Switching Loss**
- **Ids** (A): Drain current
- **Vds** (V): Drain-source voltage
- **Temperature** (°C): Junction temperature
- **Rgon** (Ω): Gate resistor for turn-on (default: 2.5 Ω)
- **Rgoff** (Ω): Gate resistor for turn-off (default: 2.5 Ω)
- **Switching Frequency** (kHz): Operating frequency
- **Duty Cycle**: PWM duty cycle (0-1)

#### **Conduction Loss (MOSFET/Body Diode)**
- **Ids** (A): Drain current
- **Temperature** (°C): Junction temperature
- **Duty Cycle**: PWM duty cycle (0-1)

#### **Thermal Impedance**
- **Time** (s): Time point for thermal impedance calculation

#### **Total Loss**
- **Ids** (A): Drain current
- **Vds** (V): Drain-source voltage (for switching)
- **Temperature** (°C): Junction temperature
- **Rgon** (Ω): Gate resistor for turn-on
- **Rgoff** (Ω): Gate resistor for turn-off
- **Switching Frequency** (kHz): Operating frequency
- **Duty Cycle**: PWM duty cycle (0-1)

---

## **Model Training**

### **Training New Models**

1. **Prepare your data:**
   - **JSON file**: Contains switching loss, conduction loss, and MOSFET characteristics
   - **Excel file**: Contains thermal impedance data (Time vs. Zth_jc)

2. **Run the training script:**
   ```bash
   python C2MSeries_ANN_full_model.py
   ```

3. **The script will:**
   - Load data from JSON and Excel files
   - Preprocess and scale the data
   - Train separate ANN models for:
     - Switching losses (Turn-On and Turn-Off)
     - Conduction losses (MOSFET and Body Diode)
     - Thermal impedance
   - Save trained models and scalers to the `models/` directory

### **Model Architecture**

The project uses a **CombinedANNModel** with separate branches for each task:

- **Switching Loss Branch**: 5 inputs (Ids, Vds, Temp, Rgon, Rgoff) → 32 → 32 → 1 output
- **Conduction MOSFET Branch**: 2 inputs (Ids, Temp) → 32 → 32 → 1 output
- **Conduction Diode Branch**: 2 inputs (Ids, Temp) → 32 → 32 → 1 output
- **Thermal Impedance Branch**: 1 input (Time) → 64 → 64 → 1 output

All branches use **ReLU activation functions** and are trained independently.

### **Testing Models**

Run the test script to validate model performance:
```bash
python C2Mseries_Full_Model_Test.py
```

---

## **Features & Visualizations**

The web interface provides:

1. **3D MOSFET Conduction Loss Plot**: Surface plot showing Vds vs. Ids and Temperature
2. **Thermal Impedance Plot**: Time-domain thermal impedance curve
3. **Switching Loss Plots**: 3D surface plots for Turn-On and Turn-Off energy losses
4. **3D Model Viewer**: Interactive 3D visualization of MOSFET package (TO247-3)
5. **Real-time Predictions**: Instant calculation results with detailed breakdowns

---

## **Technical Details**

### **Switching Loss Scaling**
The model accounts for gate resistor effects using scaling factors:
- **Turn-On Factor**: `0.0101 * Rgon + 0.4925`
- **Turn-Off Factor**: `0.0112 * Rgoff + 0.0433`

### **Power Loss Calculations**
- **Switching Loss**: `P_sw = (E_on + E_off) × f_sw × 10⁻⁶` (W)
- **Conduction Loss (MOSFET)**: `P_cond = Vds × Ids × Duty_Cycle` (W)
- **Conduction Loss (Body Diode)**: `P_cond = Vds × Ids × (1 - Duty_Cycle)` (W)
- **Total Loss**: `P_total = P_cond + P_sw` (W)

### **Data Preprocessing**
- **StandardScaler** from scikit-learn is used for feature normalization
- Separate scalers are saved for each model type (switching, conduction MOSFET, conduction diode)

---

## **File Formats**

### **JSON Data Format**
The JSON file should contain:
- `SemiconductorData`: Switching loss data (TurnOnLoss, TurnOffLoss)
- `CustomTables`: Conduction loss data (MOSFET, BodyDiode)
- `Package`: Package information (type, price)

### **Thermal Data Format**
Thermal impedance data can be provided as:
- **Excel file**: Columns for Time and Zth_jc
- **Text file**: Space-separated Time and Zth_jc values

---

## **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

---

## **License**

See the `LICENSE` file for details.

---

## **Author**

**Fulong Li**

---

## **References**

- PyTorch Documentation: https://pytorch.org/docs/
- Flask Documentation: https://flask.palletsprojects.com/
- scikit-learn Documentation: https://scikit-learn.org/

---

## **Notes**

- Ensure all model files and scalers are present in the `models/` directory before running predictions
- The models are trained on specific MOSFET data (C2M series). For other MOSFETs, retrain with appropriate data
- Temperature ranges and operating conditions should match the training data ranges for best accuracy

---

## **Future Enhancements**

- [ ] Support for additional MOSFET families
- [ ] Batch prediction capabilities
- [ ] Export prediction results to CSV/Excel
- [ ] API endpoints for programmatic access
- [ ] Model performance metrics and validation plots
- [ ] Support for dynamic thermal analysis
