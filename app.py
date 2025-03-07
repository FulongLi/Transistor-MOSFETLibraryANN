import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from flask import Flask, request, render_template
import os

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Define the CombinedANNModel
class CombinedANNModel(nn.Module):
    def __init__(self):
        super(CombinedANNModel, self).__init__()
        self.switch_fc1 = nn.Linear(in_features=5, out_features=32)
        self.switch_fc2 = nn.Linear(in_features=32, out_features=32)
        self.switch_fc3_on = nn.Linear(in_features=32, out_features=1)
        self.switch_fc3_off = nn.Linear(in_features=32, out_features=1)
        self.cond_mosfet_fc1 = nn.Linear(in_features=2, out_features=32)
        self.cond_mosfet_fc2 = nn.Linear(in_features=32, out_features=32)
        self.cond_mosfet_fc3 = nn.Linear(in_features=32, out_features=1)
        self.cond_diode_fc1 = nn.Linear(in_features=2, out_features=32)
        self.cond_diode_fc2 = nn.Linear(in_features=32, out_features=32)
        self.cond_diode_fc3 = nn.Linear(in_features=32, out_features=1)
        self.therm_fc1 = nn.Linear(in_features=1, out_features=64)
        self.therm_fc2 = nn.Linear(in_features=64, out_features=64)
        self.therm_fc3 = nn.Linear(in_features=64, out_features=1)
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


# Load model and scalers based on selected .pth file
def load_model_and_scalers(model_file):
    base_name = model_file.replace('_ANN_Trainned_Model.pth', '')
    scaler_switching_file = f'{base_name}_scaler_switching.pkl'
    scaler_conduction_mosfet_file = f'{base_name}_scaler_conduction_mosfet.pkl'
    scaler_conduction_diode_file = f'{base_name}_scaler_conduction_diode.pkl'

    model = CombinedANNModel()
    model.load_state_dict(torch.load(os.path.join('models', model_file), weights_only=True))
    model.eval()
    scaler_switching = joblib.load(os.path.join('models', scaler_switching_file))
    scaler_conduction_mosfet = joblib.load(os.path.join('models', scaler_conduction_mosfet_file))
    scaler_conduction_diode = joblib.load(os.path.join('models', scaler_conduction_diode_file))
    return model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode


# Get list of .pth model files in the models directory
def get_model_files():
    model_dir = 'models'
    if not os.path.exists(model_dir):
        return ['c2m0080120d_ANN_Trainned_Model.pth']
    return [f for f in os.listdir(model_dir) if f.endswith('_ANN_Trainned_Model.pth')]


# Load JSON data
def load_json_data(model_file):
    base_name = model_file.replace('_ANN_Trainned_Model.pth', '')
    json_file = os.path.join('models', f'{base_name}.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    return None


# Load thermal impedance data from file
def load_thermal_file_data(model_file):
    base_name = model_file.replace('_ANN_Trainned_Model.pth', '')
    thermal_file = os.path.join('models', f'{base_name}_thermal_data.txt')
    if os.path.exists(thermal_file):
        with open(thermal_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            times = []
            zth_values = []
            for line in lines:
                time, zth = map(float, line.strip().split())
                times.append(time)
                zth_values.append(zth)
            return {'times': times, 'zth': zth_values}
    return {'times': [0], 'zth': [0]}  # Fallback if file not found


# Generate 3D MOSFET conduction loss plot data (ANN as surface, JSON as scatter)
def generate_mosfet_3d_plot_data(model, scaler_conduction_mosfet, json_data):
    # ANN data (generate a grid for surface)
    currents = np.linspace(0, 80, 50)  # Finer grid for smooth surface
    temps = np.linspace(-25, 175, 50)  # Finer grid for smooth surface
    current_grid, temp_grid = np.meshgrid(currents, temps)
    ann_vds_grid = np.zeros_like(current_grid)

    for i in range(current_grid.shape[0]):
        for j in range(current_grid.shape[1]):
            ids = current_grid[i, j]
            temp = temp_grid[i, j]
            X_input = np.array([[ids, temp]])
            try:
                X_scaled = scaler_conduction_mosfet.transform(X_input)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    vds_pred = model(X_tensor, task='conduction_mosfet').item()
                ann_vds_grid[i, j] = max(vds_pred, 0)  # Ensure non-negative
            except Exception as e:
                ann_vds_grid[i, j] = 0.0  # Default to 0 if error occurs

    # Convert NumPy array to list for JSON serialization
    ann_vds_list = ann_vds_grid.tolist()

    # JSON data (scatter points)
    json_ids_values = []
    json_temp_values = []
    json_vds_values = []
    if json_data and 'CustomTables' in json_data and 'MOSFET' in json_data['CustomTables']:
        json_currents = json_data['CustomTables']['MOSFET']['XAxis']
        json_temps = json_data['CustomTables']['MOSFET']['YAxis']
        json_vds_data = json_data['CustomTables']['MOSFET']['FunctionValues']['Data']

        for temp_idx, temp in enumerate(json_temps):
            vds_row = json_vds_data[temp_idx]
            for ids_idx, ids in enumerate(json_currents):
                if ids >= 0:  # Only positive currents for MOSFET
                    json_ids_values.append(ids)
                    json_temp_values.append(temp)
                    json_vds_values.append(vds_row[ids_idx])

    return {
        'ann': {'currents': currents.tolist(), 'temps': temps.tolist(), 'vds': ann_vds_list},
        'json': {'ids': json_ids_values, 'temps': json_temp_values, 'vds': json_vds_values}
    }


# Generate thermal impedance plot data (ANN)
def generate_thermal_impedance_data(model):
    times = np.logspace(-6, 0, 50)  # 1 µs to 1 s
    zth_values = []

    for time in times:
        X_input = torch.tensor([[time]], dtype=torch.float32)
        with torch.no_grad():
            zth_pred = max(model(X_input, task='thermal').item(), 1e-6)  # Ensure positive for log scale
        zth_values.append(zth_pred)

    return {'times': times.tolist(), 'zth': zth_values}


# Initialize Flask app
app = Flask(__name__)


# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    model_files = get_model_files()
    selected_model = request.form.get('model_file',
                                      model_files[0] if model_files else 'c2m0080120d_ANN_Trainned_Model.pth')
    model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode = load_model_and_scalers(selected_model)

    json_data = load_json_data(selected_model)
    mosfet_3d_data = generate_mosfet_3d_plot_data(model, scaler_conduction_mosfet, json_data)
    ann_thermal_data = generate_thermal_impedance_data(model)
    file_thermal_data = load_thermal_file_data(selected_model)
    result = ""

    if request.method == 'POST':
        choice = int(request.form.get('choice', 0))
        duty_cycle = float(request.form.get('duty_cycle', 0.5))
        f_sw = float(request.form.get('f_sw', 20)) * 1000  # Convert kHz to Hz

        if not 0 <= duty_cycle <= 1:
            result = "Error: Duty cycle must be between 0 and 1!"
        else:
            try:
                if choice == 1:  # Switching Loss
                    ids = float(request.form['ids'])
                    vds = float(request.form['vds'])
                    temp = float(request.form['temp'])
                    rgon = float(request.form.get('rgon', 2.5))
                    rgoff = float(request.form.get('rgoff', 2.5))
                    X_input = np.array([[ids, vds, temp, rgon, rgoff]])
                    X_scaled = scaler_switching.transform(X_input)
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        turn_on_loss = model(X_tensor, task='switching_on').item()
                        turn_off_loss = model(X_tensor, task='switching_off').item()
                    p_switching = (turn_on_loss + turn_off_loss) * f_sw * 1e-6
                    result = (
                        f"🔹 Switching Loss Breakdown:\n"
                        f"  - Turn-On Energy (E_on): {turn_on_loss:.3f} µJ\n"
                        f"  - Turn-Off Energy (E_off): {turn_off_loss:.3f} µJ\n"
                        f"  - Total Energy per Cycle (E_on + E_off): {(turn_on_loss + turn_off_loss):.3f} µJ\n"
                        f"  - Switching Frequency (f_sw): {f_sw / 1000:.1f} kHz = {f_sw:.0f} Hz\n"
                        f"  - Switching Loss = (E_on + E_off) * f_sw * 1e-6 = {p_switching:.3f} W"
                    )

                elif choice == 2:  # Conduction Loss (MOSFET)
                    ids = float(request.form['ids'])
                    temp = float(request.form['temp'])
                    X_input = np.array([[ids, temp]])
                    X_scaled = scaler_conduction_mosfet.transform(X_input)
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        vds_pred = model(X_tensor, task='conduction_mosfet').item()
                    p_conduction = vds_pred * ids * duty_cycle
                    resistance = 1000 * vds_pred / ids if ids != 0 else float('inf')
                    power = vds_pred * ids
                    result = (
                        f"🔹 Conduction Loss Breakdown (MOSFET):\n"
                        f"  - Predicted Vds: {vds_pred:.3f} V\n"
                        f"  - Current (Ids): {ids:.3f} A\n"
                        f"  - Duty Cycle: {duty_cycle:.3f}\n"
                        f"  - Power without Duty Cycle = Vds * Ids = {power:.3f} W\n"
                        f"  - Conduction Loss = Vds * Ids * Duty Cycle = {p_conduction:.3f} W\n"
                        f"  - Calculated Resistance: {resistance:.3f} mΩ"
                    )

                elif choice == 3:  # Conduction Loss (Body Diode)
                    ids = float(request.form['ids'])
                    temp = float(request.form['temp'])
                    X_input = np.array([[ids, temp]])
                    X_scaled = scaler_conduction_diode.transform(X_input)
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        vds_pred = model(X_tensor, task='conduction_diode').item()
                    p_conduction = vds_pred * ids * (1 - duty_cycle)
                    resistance = 1000 * vds_pred / ids if ids != 0 else float('inf')
                    power = vds_pred * ids
                    result = (
                        f"🔹 Conduction Loss Breakdown (Body Diode):\n"
                        f"  - Predicted Vds: {vds_pred:.3f} V\n"
                        f"  - Current (Ids): {ids:.3f} A\n"
                        f"  - Duty Cycle (off-time): {1 - duty_cycle:.3f}\n"
                        f"  - Power without Duty Cycle = Vds * Ids = {power:.3f} W\n"
                        f"  - Conduction Loss = Vds * Ids * (1 - Duty Cycle) = {p_conduction:.3f} W\n"
                        f"  - Calculated Resistance: {resistance:.3f} mΩ"
                    )

                elif choice == 4:  # Thermal Impedance
                    time = float(request.form['time'])
                    X_input = torch.tensor([[time]], dtype=torch.float32)
                    with torch.no_grad():
                        zth_pred = model(X_input, task='thermal').item()
                    result = (
                        f"🔹 Thermal Impedance Breakdown:\n"
                        f"  - Predicted Zth_jc at {time:.3f} s: {zth_pred:.6f} K/W"
                    )

                elif choice == 5:  # Total Loss
                    ids = float(request.form['ids'])
                    temp = float(request.form['temp'])
                    vds_switch = float(request.form['vds'])
                    rgon = float(request.form.get('rgon', 2.5))
                    rgoff = float(request.form.get('rgoff', 2.5))
                    X_input_mosfet = np.array([[ids, temp]])
                    X_scaled_mosfet = scaler_conduction_mosfet.transform(X_input_mosfet)
                    X_tensor_mosfet = torch.tensor(X_scaled_mosfet, dtype=torch.float32)
                    with torch.no_grad():
                        vds_mosfet = model(X_tensor_mosfet, task='conduction_mosfet').item()
                    p_conduction_mosfet = vds_mosfet * ids * duty_cycle
                    X_input_switch = np.array([[ids, vds_switch, temp, rgon, rgoff]])
                    X_scaled_switch = scaler_switching.transform(X_input_switch)
                    X_tensor_switch = torch.tensor(X_scaled_switch, dtype=torch.float32)
                    with torch.no_grad():
                        turn_on_loss = model(X_tensor_switch, task='switching_on').item()
                        turn_off_loss = model(X_tensor_switch, task='switching_off').item()
                    p_switching = (turn_on_loss + turn_off_loss) * f_sw * 1e-6
                    p_total = p_conduction_mosfet + p_switching
                    result = (
                        f"🔹 Total Power Loss Breakdown (excluding Body Diode):\n"
                        f"  - MOSFET Conduction Loss:\n"
                        f"    - Predicted Vds (MOSFET): {vds_mosfet:.3f} V\n"
                        f"    - Current (Ids): {ids:.3f} A\n"
                        f"    - Duty Cycle: {duty_cycle:.3f}\n"
                        f"    - Power without Duty Cycle = Vds * Ids = {vds_mosfet * ids:.3f} W\n"
                        f"    - Conduction Loss = Vds * Ids * Duty Cycle = {p_conduction_mosfet:.3f} W\n"
                        f"  - Switching Loss:\n"
                        f"    - Turn-On Energy (E_on): {turn_on_loss:.3f} µJ\n"
                        f"    - Turn-Off Energy (E_off): {turn_off_loss:.3f} µJ\n"
                        f"    - Total Energy per Cycle (E_on + E_off): {(turn_on_loss + turn_off_loss):.3f} µJ\n"
                        f"    - Switching Frequency (f_sw): {f_sw / 1000:.1f} kHz = {f_sw:.0f} Hz\n"
                        f"    - Switching Loss = (E_on + E_off) * f_sw * 1e-6 = {p_switching:.3f} W\n"
                        f"  - Total Loss = Conduction Loss + Switching Loss = {p_total:.3f} W"
                    )

                else:
                    result = "⚠️ Invalid choice! Please select 1, 2, 3, 4, or 5."

            except Exception as e:
                result = f"⚠️ Error: {str(e)}"

    return render_template('index.html', result=result, mosfet_3d_data=mosfet_3d_data,
                           ann_thermal_data=ann_thermal_data, file_thermal_data=file_thermal_data,
                           model_files=model_files, selected_model=selected_model)


if __name__ == '__main__':
    app.run(debug=True)