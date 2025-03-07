import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# Set Random Seed for Reproducibility
# --------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------------------
# Utility Functions
# --------------------------
def load_json_data(file_path='C2M0080120D_ANN.json'):
    """Load JSON data."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_excel_data(file_path='C2M0040120D_Thermal_Impedance_Data.xlsx'):
    """Load Excel data."""
    df = pd.read_excel(file_path)
    if 'Time' in df.columns and 'Junction to Case Impedance Zth(J2C)' in df.columns:
        x_np = df['Time'].values.astype(np.float32)
        y_np = df['Junction to Case Impedance Zth(J2C)'].values.astype(np.float32)
    else:
        x_np = df.iloc[:, 0].values.astype(np.float32)
        y_np = df.iloc[:, 1].values.astype(np.float32)
    return x_np.reshape(-1, 1), y_np.reshape(-1, 1)


def generate_switching_loss_data(turn_data, label, rgon_range, rgoff_range):
    """Generate switching loss data."""
    current_axis_loss = np.array(turn_data['CurrentAxis'], dtype=float)
    voltage_axis_loss = np.array(turn_data['VoltageAxis'], dtype=float)
    energy_data = turn_data['Energy']['Data']
    LossData = []
    index = 0
    for temp_dict in energy_data:
        temp = temp_dict['Temperature']
        for v_idx, voltage in enumerate(voltage_axis_loss):
            if voltage < 0:
                continue
            values = temp_dict['Voltages'][v_idx]['Values']
            for c_idx, energy_value in enumerate(values):
                current_val = current_axis_loss[c_idx]
                for Rgon in rgon_range:
                    for Rgoff in rgoff_range:
                        turn_on_factor = 0.0101 * Rgon + 0.4925
                        turn_off_factor = 0.0112 * Rgoff + 0.0433
                        base_turn_on = 0.0101 * 2.5 + 0.4925
                        base_turn_off = 0.0112 * 2.5 + 0.0433
                        energy_scaled = (energy_value / base_turn_on) * turn_on_factor if label == "TurnOnEnergy" else \
                            (energy_value / base_turn_off) * turn_off_factor
                        LossData.append([index, temp, voltage, current_val, Rgon, Rgoff, energy_scaled])
                        index += 1
    return pd.DataFrame(LossData, columns=["Index", "Temperature", "Voltage", "Current", "Rgon", "Rgoff", label])


def generate_conduction_loss_data(json_data):
    """Generate conduction loss data for MOSFET and BodyDiode."""
    # MOSFET Conduction Loss (CustomTables)
    mosfet_data = json_data['CustomTables']['MOSFET']
    current_axis_mosfet = np.array(mosfet_data['XAxis'])
    temperature_axis_mosfet = np.array(mosfet_data['YAxis'])
    voltage_drop_data_mosfet = mosfet_data['FunctionValues']['Data']
    ConductionLossData_MOSFET = []
    index = 0
    for t_idx, temp in enumerate(temperature_axis_mosfet):
        for i_idx, ids in enumerate(current_axis_mosfet):
            vds = voltage_drop_data_mosfet[t_idx][i_idx]
            ConductionLossData_MOSFET.append([index, vds, ids, temp])
            index += 1

    # BodyDiode Conduction Loss (CustomTables)
    diode_data = json_data['CustomTables']['BodyDiode']
    current_axis_diode = np.array(diode_data['XAxis'])
    temperature_axis_diode = np.array(diode_data['YAxis'])
    voltage_drop_data_diode = diode_data['FunctionValues']['Data']
    ConductionLossData_Diode = []
    index = 0
    for t_idx, temp in enumerate(temperature_axis_diode):
        for i_idx, ids in enumerate(current_axis_diode):
            vds = voltage_drop_data_diode[t_idx][i_idx]
            ConductionLossData_Diode.append([index, vds, ids, temp])
            index += 1

    return (pd.DataFrame(ConductionLossData_MOSFET, columns=["Index", "Vds", "Ids", "Temperature"]),
            pd.DataFrame(ConductionLossData_Diode, columns=["Index", "Vds", "Ids", "Temperature"]))


def prepare_data(df, feature_cols, target_col):
    """Prepare data for training with scaling."""
    X = df[feature_cols].values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1)), scaler, X_test, y_test


def prepare_thermal_data(x_np, y_np):
    """Prepare thermal data (no scaling needed)."""
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


def train_model(model, optimizer, datasets, epochs, name=""):
    """Train the combined ANN model."""
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        # Switching On
        outputs_on = model(datasets['switching_on'][0], task='switching_on')
        loss_on = criterion(outputs_on, datasets['switching_on'][1])
        total_loss += loss_on
        # Switching Off
        outputs_off = model(datasets['switching_off'][0], task='switching_off')
        loss_off = criterion(outputs_off, datasets['switching_off'][1])
        total_loss += loss_off
        # Conduction (MOSFET)
        outputs_conduction_mosfet = model(datasets['conduction_mosfet'][0], task='conduction_mosfet')
        loss_conduction_mosfet = criterion(outputs_conduction_mosfet, datasets['conduction_mosfet'][1])
        total_loss += loss_conduction_mosfet
        # Conduction (BodyDiode)
        outputs_conduction_diode = model(datasets['conduction_diode'][0], task='conduction_diode')
        loss_conduction_diode = criterion(outputs_conduction_diode, datasets['conduction_diode'][1])
        total_loss += loss_conduction_diode
        # Thermal
        outputs_thermal = model(datasets['thermal'][0], task='thermal')
        loss_thermal = criterion(outputs_thermal, datasets['thermal'][1])
        total_loss += loss_thermal

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % (epochs // 10) == 0:
            print(f"{name} Epoch [{epoch + 1}/{epochs}], Total Loss: {total_loss.item():.6f} "
                  f"(On: {loss_on.item():.6f}, Off: {loss_off.item():.6f}, MOSFET: {loss_conduction_mosfet.item():.6f}, "
                  f"Diode: {loss_conduction_diode.item():.6f}, Thermal: {loss_thermal.item():.6f})")


# --------------------------
# Combined ANN Model Definition
# --------------------------
class CombinedANNModel(nn.Module):
    def __init__(self):
        super(CombinedANNModel, self).__init__()
        # Switching layers (5 inputs: Current, Voltage, Temperature, Rgon, Rgoff)
        self.switch_fc1 = nn.Linear(5, 16)
        self.switch_fc2 = nn.Linear(16, 16)
        self.switch_fc3_on = nn.Linear(16, 1)  # Turn-On output
        self.switch_fc3_off = nn.Linear(16, 1)  # Turn-Off output
        # Conduction MOSFET layers (2 inputs: Ids, Temperature)
        self.cond_mosfet_fc1 = nn.Linear(2, 25)
        self.cond_mosfet_fc2 = nn.Linear(25, 16)
        self.cond_mosfet_fc3 = nn.Linear(16, 1)
        # Conduction BodyDiode layers (2 inputs: Ids, Temperature)
        self.cond_diode_fc1 = nn.Linear(2, 25)
        self.cond_diode_fc2 = nn.Linear(25, 16)
        self.cond_diode_fc3 = nn.Linear(16, 1)
        # Thermal layers (1 input: Time)
        self.therm_fc1 = nn.Linear(1, 64)
        self.therm_fc2 = nn.Linear(64, 64)
        self.therm_fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x, task='switching_on'):
        if task == 'switching_on':
            x = self.relu(self.switch_fc1(x))
            x = self.relu(self.switch_fc2(x))
            return self.switch_fc3_on(x)
        elif task == 'switching_off':
            x = self.relu(self.switch_fc1(x))
            x = self.relu(self.switch_fc2(x))
            return self.switch_fc3_off(x)
        elif task == 'conduction_mosfet':
            x = self.relu(self.cond_mosfet_fc1(x))
            x = self.relu(self.cond_mosfet_fc2(x))
            return self.cond_mosfet_fc3(x)
        elif task == 'conduction_diode':
            x = self.relu(self.cond_diode_fc1(x))
            x = self.relu(self.cond_diode_fc2(x))
            return self.cond_diode_fc3(x)
        elif task == 'thermal':
            x = self.relu(self.therm_fc1(x))
            x = self.relu(self.therm_fc2(x))
            return self.therm_fc3(x)
        else:
            raise ValueError("Invalid task specified")


# --------------------------
# Process and Train Combined Model
# --------------------------
def process_and_train_combined_model(json_data, excel_x_np, excel_y_np):
    # Switching Loss Data
    rgon_range = np.arange(1, 31, 1)
    rgoff_range = np.arange(1, 31, 1)
    TurnOnLossData = generate_switching_loss_data(json_data['SemiconductorData']['TurnOnLoss'], "TurnOnEnergy",
                                                  rgon_range, rgoff_range)
    TurnOffLossData = generate_switching_loss_data(json_data['SemiconductorData']['TurnOffLoss'], "TurnOffEnergy",
                                                   rgon_range, rgoff_range)
    feature_cols_switch = ['Current', 'Voltage', 'Temperature', 'Rgon', 'Rgoff']
    (X_on_train, y_on_train, X_on_test, y_on_test), scaler_switching, _, _ = prepare_data(TurnOnLossData,
                                                                                          feature_cols_switch,
                                                                                          'TurnOnEnergy')
    (X_off_train, y_off_train, X_off_test, y_off_test), _, _, _ = prepare_data(TurnOffLossData, feature_cols_switch,
                                                                               'TurnOffEnergy')

    # Conduction Loss Data (MOSFET and BodyDiode)
    ConductionLossData_MOSFET, ConductionLossData_Diode = generate_conduction_loss_data(json_data)
    feature_cols_cond = ['Ids', 'Temperature']
    (X_cond_mosfet_train, y_cond_mosfet_train, X_cond_mosfet_test,
     y_cond_mosfet_test), scaler_conduction_mosfet, X_cond_mosfet_test_raw, y_cond_mosfet_test_raw = prepare_data(
        ConductionLossData_MOSFET, feature_cols_cond, 'Vds')
    (X_cond_diode_train, y_cond_diode_train, X_cond_diode_test,
     y_cond_diode_test), scaler_conduction_diode, X_cond_diode_test_raw, y_cond_diode_test_raw = prepare_data(
        ConductionLossData_Diode, feature_cols_cond, 'Vds')

    # Thermal Impedance Data
    X_therm_train, y_therm_train = prepare_thermal_data(excel_x_np, excel_y_np)

    # Combine datasets for training
    datasets = {
        'switching_on': (X_on_train, y_on_train),
        'switching_off': (X_off_train, y_off_train),
        'conduction_mosfet': (X_cond_mosfet_train, y_cond_mosfet_train),
        'conduction_diode': (X_cond_diode_train, y_cond_diode_train),
        'thermal': (X_therm_train, y_therm_train)
    }

    # Instantiate and train the combined model
    model = CombinedANNModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, optimizer, datasets, 2000, "Combined")

    # Automatic save with updated file name
    model_file_name = os.path.splitext('c2m0080120d.json')[0] + '_ANN_Trainned_Model.pth'
    torch.save(model.state_dict(), model_file_name)
    scaler_switching_file = os.path.splitext('c2m0080120d.json')[0] + '_scaler_switching.pkl'
    scaler_conduction_mosfet_file = os.path.splitext('c2m0080120d.json')[0] + '_scaler_conduction_mosfet.pkl'
    scaler_conduction_diode_file = os.path.splitext('c2m0080120d.json')[0] + '_scaler_conduction_diode.pkl'
    joblib.dump(scaler_switching, scaler_switching_file)
    joblib.dump(scaler_conduction_mosfet, scaler_conduction_mosfet_file)
    joblib.dump(scaler_conduction_diode, scaler_conduction_diode_file)

    print(f"✅ Combined Model trained and saved as '{model_file_name}'!")

    return (model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode,
            ConductionLossData_MOSFET, X_cond_mosfet_test_raw, y_cond_mosfet_test_raw,
            ConductionLossData_Diode, X_cond_diode_test_raw, y_cond_diode_test_raw,
            excel_x_np, excel_y_np)


# --------------------------
# Visualization Functions
# --------------------------
def plot_conduction_results(model, scaler, ConductionLossData, X_test_raw, y_test_raw, task='conduction_mosfet'):
    X_test_scaled = scaler.transform(X_test_raw)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_tensor, task=task).numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_raw, y_test_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], 'r--', label='Ideal Fit')
    plt.xlabel('Actual Vds (V)')
    plt.ylabel('Predicted Vds (V)')
    plt.title(f'Actual vs Predicted Vds ({task.replace("_", " ").title()})')
    plt.legend()
    plt.grid(True)
    plt.show()

    ids_range = np.linspace(ConductionLossData['Ids'].min(), ConductionLossData['Ids'].max(), 30)
    temp_range = np.linspace(ConductionLossData['Temperature'].min(), ConductionLossData['Temperature'].max(), 30)
    ids_grid, temp_grid = np.meshgrid(ids_range, temp_range)
    X_grid = np.column_stack([ids_grid.ravel(), temp_grid.ravel()])
    X_grid_scaled = scaler.transform(X_grid)
    X_grid_tensor = torch.tensor(X_grid_scaled, dtype=torch.float32)

    with torch.no_grad():
        vds_grid_pred = model(X_grid_tensor, task=task).numpy().reshape(ids_grid.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(ConductionLossData['Ids'], ConductionLossData['Temperature'], ConductionLossData['Vds'],
                         c=ConductionLossData['Temperature'], cmap='viridis', marker='o', alpha=0.6,
                         label='Data Points')
    ax.plot_surface(ids_grid, temp_grid, vds_grid_pred, color='blue', alpha=0.3)
    ax.set_xlabel('Ids (A)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_zlabel('Vds (V)')
    ax.set_title(f'ANN Predictions for {task.replace("_", " ").title()}')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Temperature (°C)')
    plt.legend()
    plt.show()


def plot_thermal_results(model, x_np, y_np):
    x_dense = np.linspace(x_np.min(), x_np.max(), 300).reshape(-1, 1).astype(np.float32)
    x_dense_tensor = torch.from_numpy(x_dense)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_dense_tensor, task='thermal').numpy()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].scatter(x_np, y_np, color='blue', label='Data')
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Zth_jc (K/W)")
    axs[0, 0].set_title("Original Data (Linear Scale)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].scatter(x_np, y_np, color='blue', label='Data')
    axs[0, 1].set_xlabel("Time (log scale)")
    axs[0, 1].set_ylabel("Zth_jc (log scale)")
    axs[0, 1].set_title("Original Data (Log-Log Scale)")
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].scatter(x_np, y_np, color='blue', label='Data')
    axs[1, 0].plot(x_dense, y_pred, color='red', label='ANN Fit')
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Zth_jc (K/W)")
    axs[1, 0].set_title("ANN Fit (Linear Scale)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].scatter(x_np, y_np, color='blue', label='Data')
    axs[1, 1].plot(x_dense, y_pred, color='red', label='ANN Fit')
    axs[1, 1].set_xlabel("Time (log scale)")
    axs[1, 1].set_ylabel("Zth_jc (log scale)")
    axs[1, 1].set_title("ANN Fit (Log-Log Scale)")
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------
# Combined Prediction Loop
# --------------------------
def predict_all(model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode):
    while True:
        try:
            choice = input(
                "\nEnter 1 for Switching loss, 2 for Conduction loss for MOSFET, 3 for Conduction loss for BodyDiode, 4 for Thermal Impedance, 5 for Exit: ").strip().lower()
            choice = choice if choice.isdigit() else '0'  # Ensure choice is a number, default to 0 if not
            choice = int(choice)
            if choice == 5:
                break
            elif choice == 1:
                ids = float(input("Enter Current (Ids) in A: "))
                vds = float(input("Enter Voltage (Vds) in V: "))
                temp = float(input("Enter Temperature in °C: "))
                rgon = float(input("Enter Rgon in Ω (default 2.5): ") or 2.5)
                rgoff = float(input("Enter Rgoff in Ω (default 2.5): ") or 2.5)
                X_input = np.array([[ids, vds, temp, rgon, rgoff]])
                X_scaled = scaler_switching.transform(X_input)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    turn_on_loss = model(X_tensor, task='switching_on').item()
                    turn_off_loss = model(X_tensor, task='switching_off').item()
                print(f"\n🔹 Turn-On Loss: {turn_on_loss:.3f} µJ")
                print(f"🔹 Turn-Off Loss: {turn_off_loss:.3f} µJ")
            elif choice == 2:
                ids = float(input("Enter Ids in A: "))
                temp = float(input("Enter Temperature in °C: "))
                X_input = np.array([[ids, temp]])
                X_scaled = scaler_conduction_mosfet.transform(X_input)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    vds_pred = model(X_tensor, task='conduction_mosfet').item()
                resistance = 1000 * vds_pred / ids if ids != 0 else float('inf')
                power = vds_pred * ids
                print(f"\n🔹 Predicted Vds (Conduction loss for MOSFET): {vds_pred:.3f} V")
                print(f"🔹 Calculated Resistance: {resistance:.3f} mΩ")
                print(f"🔹 Calculated Power: {power:.3f} W")
            elif choice == 3:
                ids = float(input("Enter Ids in A: "))
                temp = float(input("Enter Temperature in °C: "))
                X_input = np.array([[ids, temp]])
                X_scaled = scaler_conduction_diode.transform(X_input)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    vds_pred = model(X_tensor, task='conduction_diode').item()
                resistance = 1000 * vds_pred / ids if ids != 0 else float('inf')
                power = vds_pred * ids
                print(f"\n🔹 Predicted Vds (Conduction loss for BodyDiode): {vds_pred:.3f} V")
                print(f"🔹 Calculated Resistance: {resistance:.3f} mΩ")
                print(f"🔹 Calculated Power: {power:.3f} W")
            elif choice == 4:
                time = float(input("Enter Time in seconds: "))
                X_input = torch.tensor([[time]], dtype=torch.float32)
                with torch.no_grad():
                    zth_pred = model(X_input, task='thermal').item()
                print(f"\n🔹 Predicted Thermal Impedance Zth_jc: {zth_pred:.6f} K/W")
            else:
                print("⚠️ Invalid choice! Enter 1, 2, 3, 4, or 5.")
        except ValueError as e:
            print(f"⚠️ Invalid input! Error: {e}")
        except Exception as e:
            print(f"⚠️ An error occurred: {e}")


# --------------------------
# Execute Combined Code
# --------------------------
if __name__ == "__main__":
    print("Loading Data...")
    json_data = load_json_data()
    excel_x_np, excel_y_np = load_excel_data()

    print("Processing and Training Combined Model...")
    (model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode,
     ConductionLossData_MOSFET, X_cond_mosfet_test_raw, y_cond_mosfet_test_raw,
     ConductionLossData_Diode, X_cond_diode_test_raw, y_cond_diode_test_raw,
     excel_x_np, excel_y_np) = process_and_train_combined_model(json_data, excel_x_np, excel_y_np)
    print("✅ Combined Model trained and saved as 'c2m0080120d_ANN_Trainned_Model.pth'!")

    print("Generating Visualizations...")
    plot_conduction_results(model, scaler_conduction_mosfet, ConductionLossData_MOSFET, X_cond_mosfet_test_raw,
                            y_cond_mosfet_test_raw, task='conduction_mosfet')
    plot_conduction_results(model, scaler_conduction_diode, ConductionLossData_Diode, X_cond_diode_test_raw,
                            y_cond_diode_test_raw, task='conduction_diode')
    plot_thermal_results(model, excel_x_np, excel_y_np)

    print("Starting Prediction Loop...")
    predict_all(model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode)