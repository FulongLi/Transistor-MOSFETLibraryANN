import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
import os

# --------------------------
# Set Random Seed for Reproducibility
# --------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
# Load Saved Model and Scalers
# --------------------------
def load_model_and_scalers():
    # Define file names based on the JSON file name
    model_file_name = os.path.splitext('c2m0080120d.json')[0] + '_ANN_Trainned_Model.pth'
    scaler_switching_file = os.path.splitext('c2m0080120d.json')[0] + '_scaler_switching.pkl'
    scaler_conduction_mosfet_file = os.path.splitext('c2m0080120d.json')[0] + '_scaler_conduction_mosfet.pkl'
    scaler_conduction_diode_file = os.path.splitext('c2m0080120d.json')[0] + '_scaler_conduction_diode.pkl'

    # Load the model with weights_only=True
    model = CombinedANNModel()
    model.load_state_dict(torch.load(model_file_name, weights_only=True))
    model.eval()  # Set model to evaluation mode

    # Load the scalers
    scaler_switching = joblib.load(scaler_switching_file)
    scaler_conduction_mosfet = joblib.load(scaler_conduction_mosfet_file)
    scaler_conduction_diode = joblib.load(scaler_conduction_diode_file)

    print(f"✅ Model loaded from '{model_file_name}'!")
    print("✅ Scalers loaded successfully!")

    return model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode


# --------------------------
# Combined Prediction Loop with Loss Calculation and Breakdown
# --------------------------
def predict_all(model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode):
    while True:
        try:
            choice = input(
                "\nEnter 1 for Switching loss, 2 for Conduction loss for MOSFET, 3 for Conduction loss for BodyDiode, 4 for Thermal Impedance, 5 for Total Loss, 6 for Exit: ").strip().lower()
            choice = choice if choice.isdigit() else '0'  # Ensure choice is a number, default to 0 if not
            choice = int(choice)
            if choice == 6:
                break

            # Prompt for common parameters (duty cycle and switching frequency)
            duty_cycle = float(input("Enter duty cycle (0 to 1): ") or 0.5)  # Default to 0.5 if not provided
            if not 0 <= duty_cycle <= 1:
                raise ValueError("Duty cycle must be between 0 and 1!")
            f_sw = float(input("Enter switching frequency (kHz): ") or 20)  # Default to 20 kHz if not provided
            f_sw = f_sw * 1000  # Convert kHz to Hz

            if choice == 1:
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
                p_switching = (turn_on_loss + turn_off_loss) * f_sw * 1e-6  # Convert µJ to J and multiply by frequency
                print("\n🔹 Switching Loss Breakdown:")
                print(f"  - Turn-On Energy (E_on): {turn_on_loss:.3f} µJ")
                print(f"  - Turn-Off Energy (E_off): {turn_off_loss:.3f} µJ")
                print(f"  - Total Energy per Cycle (E_on + E_off): {(turn_on_loss + turn_off_loss):.3f} µJ")
                print(f"  - Switching Frequency (f_sw): {f_sw / 1000:.1f} kHz = {f_sw:.0f} Hz")
                print(
                    f"  - Switching Loss = (E_on + E_off) * f_sw * 1e-6 = {(turn_on_loss + turn_off_loss):.3f} * {f_sw:.0f} * 1e-6 = {p_switching:.3f} W")

            elif choice == 2:
                ids = float(input("Enter Ids in A: "))
                temp = float(input("Enter Temperature in °C: "))
                X_input = np.array([[ids, temp]])
                X_scaled = scaler_conduction_mosfet.transform(X_input)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    vds_pred = model(X_tensor, task='conduction_mosfet').item()
                p_conduction = vds_pred * ids * duty_cycle
                resistance = 1000 * vds_pred / ids if ids != 0 else float('inf')
                power = vds_pred * ids
                print("\n🔹 Conduction Loss Breakdown (MOSFET):")
                print(f"  - Predicted Vds: {vds_pred:.3f} V")
                print(f"  - Current (Ids): {ids:.3f} A")
                print(f"  - Duty Cycle: {duty_cycle:.3f}")
                print(f"  - Power without Duty Cycle = Vds * Ids = {vds_pred:.3f} * {ids:.3f} = {power:.3f} W")
                print(
                    f"  - Conduction Loss = Vds * Ids * Duty Cycle = {vds_pred:.3f} * {ids:.3f} * {duty_cycle:.3f} = {p_conduction:.3f} W")
                print(f"  - Calculated Resistance: {resistance:.3f} mΩ")

            elif choice == 3:
                ids = float(input("Enter Ids in A: "))
                temp = float(input("Enter Temperature in °C: "))
                X_input = np.array([[ids, temp]])
                X_scaled = scaler_conduction_diode.transform(X_input)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    vds_pred = model(X_tensor, task='conduction_diode').item()
                p_conduction = vds_pred * ids * (1 - duty_cycle)  # BodyDiode conducts during off-time
                resistance = 1000 * vds_pred / ids if ids != 0 else float('inf')
                power = vds_pred * ids
                print("\n🔹 Conduction Loss Breakdown (BodyDiode):")
                print(f"  - Predicted Vds: {vds_pred:.3f} V")
                print(f"  - Current (Ids): {ids:.3f} A")
                print(f"  - Duty Cycle (off-time): {1 - duty_cycle:.3f}")
                print(f"  - Power without Duty Cycle = Vds * Ids = {vds_pred:.3f} * {ids:.3f} = {power:.3f} W")
                print(
                    f"  - Conduction Loss = Vds * Ids * (1 - Duty Cycle) = {vds_pred:.3f} * {ids:.3f} * {1 - duty_cycle:.3f} = {p_conduction:.3f} W")
                print(f"  - Calculated Resistance: {resistance:.3f} mΩ")

            elif choice == 4:
                time = float(input("Enter Time in seconds: "))
                X_input = torch.tensor([[time]], dtype=torch.float32)
                with torch.no_grad():
                    zth_pred = model(X_input, task='thermal').item()
                print(f"\n🔹 Thermal Impedance Breakdown:")
                print(f"  - Predicted Zth_jc at {time:.3f} s: {zth_pred:.6f} K/W")

            elif choice == 5:
                # Calculate total loss by combining MOSFET conduction with switching loss (excluding BodyDiode)
                # Shared inputs for MOSFET and Switching
                ids = float(input("Enter Ids in A: "))
                temp = float(input("Enter Temperature in °C: "))
                vds_switch = float(input("Enter Voltage (Vds) for Switching in V: "))
                rgon = float(input("Enter Rgon in Ω (default 2.5): ") or 2.5)
                rgoff = float(input("Enter Rgoff in Ω (default 2.5): ") or 2.5)

                # MOSFET Conduction
                X_input_mosfet = np.array([[ids, temp]])
                X_scaled_mosfet = scaler_conduction_mosfet.transform(X_input_mosfet)
                X_tensor_mosfet = torch.tensor(X_scaled_mosfet, dtype=torch.float32)
                with torch.no_grad():
                    vds_mosfet = model(X_tensor_mosfet, task='conduction_mosfet').item()
                p_conduction_mosfet = vds_mosfet * ids * duty_cycle

                # Switching Loss
                X_input_switch = np.array([[ids, vds_switch, temp, rgon, rgoff]])
                X_scaled_switch = scaler_switching.transform(X_input_switch)
                X_tensor_switch = torch.tensor(X_scaled_switch, dtype=torch.float32)
                with torch.no_grad():
                    turn_on_loss = model(X_tensor_switch, task='switching_on').item()
                    turn_off_loss = model(X_tensor_switch, task='switching_off').item()
                p_switching = (turn_on_loss + turn_off_loss) * f_sw * 1e-6  # Convert µJ to J and multiply by frequency

                # Total Loss (MOSFET conduction + Switching, excluding BodyDiode)
                p_total = p_conduction_mosfet + p_switching

                print("\n🔹 Total Power Loss Breakdown (excluding BodyDiode):")
                print(f"  - MOSFET Conduction Loss:")
                print(f"    - Predicted Vds (MOSFET): {vds_mosfet:.3f} V")
                print(f"    - Current (Ids): {ids:.3f} A")
                print(f"    - Duty Cycle: {duty_cycle:.3f}")
                print(
                    f"    - Power without Duty Cycle = Vds * Ids = {vds_mosfet:.3f} * {ids:.3f} = {vds_mosfet * ids:.3f} W")
                print(
                    f"    - Conduction Loss = Vds * Ids * Duty Cycle = {vds_mosfet:.3f} * {ids:.3f} * {duty_cycle:.3f} = {p_conduction_mosfet:.3f} W")
                print(f"  - Switching Loss:")
                print(f"    - Turn-On Energy (E_on): {turn_on_loss:.3f} µJ")
                print(f"    - Turn-Off Energy (E_off): {turn_off_loss:.3f} µJ")
                print(f"    - Total Energy per Cycle (E_on + E_off): {(turn_on_loss + turn_off_loss):.3f} µJ")
                print(f"    - Switching Frequency (f_sw): {f_sw / 1000:.1f} kHz = {f_sw:.0f} Hz")
                print(
                    f"    - Switching Loss = (E_on + E_off) * f_sw * 1e-6 = {(turn_on_loss + turn_off_loss):.3f} * {f_sw:.0f} * 1e-6 = {p_switching:.3f} W")
                print(
                    f"  - Total Loss = Conduction Loss + Switching Loss = {p_conduction_mosfet:.3f} + {p_switching:.3f} = {p_total:.3f} W")

            else:
                print("⚠️ Invalid choice! Enter 1, 2, 3, 4, 5, or 6.")

        except ValueError as e:
            print(f"⚠️ Invalid input! Error: {e}")
        except Exception as e:
            print(f"⚠️ An error occurred: {e}")


# --------------------------
# Execute Code to Load Model and Run Prediction
# --------------------------
if __name__ == "__main__":
    # Load the saved model and scalers
    model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode = load_model_and_scalers()

    # Start the prediction loop
    print("Starting Prediction Loop...")
    predict_all(model, scaler_switching, scaler_conduction_mosfet, scaler_conduction_diode)