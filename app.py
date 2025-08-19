# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import smtplib
import hashlib
import hmac
from email.mime.text import MIMEText
from email.header import Header
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- CONFIG ----------------
USERS_CSV = "users.csv"
MODEL_FILE = "health_model.pkl"       # Your trained ML model
HISTORY_PREFIX = "history_"           # Stored as history_<username>.csv

# ---------------- PRECAUTIONS ----------------
precautions = {
    "Heart Rate": "If too high, rest and avoid stress. If too low, seek medical advice.",
    "Respiratory Rate": "Practice calm breathing; if too high/low, consult a doctor.",
    "Body Temperature": "If feverish, keep hydrated and rest. If too low, keep warm.",
    "Oxygen Saturation": "If below 95%, use breathing exercises. If <90%, seek urgent care.",
    "Systolic Blood Pressure": "High: reduce salt & stress. Low: increase fluids & rest.",
    "Diastolic Blood Pressure": "High: avoid stress, follow a low-salt diet. Low: hydrate well.",
    "Derived_BMI": "Maintain balanced diet and regular exercise.",
    "Derived_MAP": "Abnormal MAP may need medical evaluation. Monitor closely."
}

# ---------------- MODEL ----------------
def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Please add your trained model.")
        st.stop()
    return joblib.load(path)

# ---------------- PDF GENERATOR ----------------
def generate_pdf_report_line_by_line(df, patient_name, username):
    from io import BytesIO
    def draw_header(c, width, height):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Patient Health Report")
        c.setFont("Helvetica", 11)
        c.drawString(50, height - 75, f"Patient Name: {patient_name}")
        c.drawString(50, height - 90, f"Username: {username}")
        c.drawString(50, height - 105, f"Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    c.setTitle(f"report_{username}")
    draw_header(c, width, height)

    y = height - 140
    line_h = 14

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    for _, row in df.iterrows():
        if y < 90:
            c.showPage()
            draw_header(c, width, height)
            y = height - 70
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Date: {row['Date']}")
        y -= line_h
        c.setFont("Helvetica", 11)
        for col in row.index:
            if col != "Date":
                val = row[col]
                if isinstance(val, float):
                    val = int(val) if val.is_integer() else round(val, 2)
                c.drawString(70, y, f"{col}: {val}")
                y -= line_h
        y -= 6

    c.save()
    buf.seek(0)
    return buf

# ---------------- SECURITY ----------------
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return salt.hex() + ":" + key.hex()

def verify_password(stored: str, provided: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        stored_key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", provided.encode("utf-8"), salt, 100000)
        return hmac.compare_digest(new_key, stored_key)
    except Exception:
        return False

# ---------------- USER DB ----------------
def ensure_users_csv():
    if not os.path.exists(USERS_CSV):
        df = pd.DataFrame(columns=["username", "name", "password", "caretaker_email"])
        df.to_csv(USERS_CSV, index=False)

def read_users():
    ensure_users_csv()
    return pd.read_csv(USERS_CSV)

def add_user(username, name, hashed_password, caretaker_email):
    df = read_users()
    if username in df["username"].values:
        return False, "Username already exists"
    df = pd.concat([df, pd.DataFrame([[username, name, hashed_password, caretaker_email]], columns=df.columns)], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True, "User added"

# ---------------- HISTORY ----------------
def save_history(username, row_dict):
    filename = f"{HISTORY_PREFIX}{username}.csv"
    if os.path.exists(filename):
        hdf = pd.read_csv(filename)
    else:
        hdf = pd.DataFrame()
    hdf = pd.concat([hdf, pd.DataFrame([row_dict])], ignore_index=True)
    hdf.to_csv(filename, index=False)

def load_history(username):
    filename = f"{HISTORY_PREFIX}{username}.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

# ---------------- EMAIL ----------------
def get_email_credentials():
    try:
        sender = st.secrets.get("EMAIL_SENDER", None)
        app_pw = st.secrets.get("EMAIL_PASSWORD", None)
    except Exception:
        sender = os.getenv("EMAIL_SENDER")
        app_pw = os.getenv("EMAIL_PASSWORD")
    if sender and app_pw:
        return sender, app_pw
    return "jadhavdhanashri0602@gmail.com", "qoxg peuy nkwo crcx"

def send_email_alert(patient_data: dict, receiver_email: str):
    sender_email, sender_password = get_email_credentials()
    if not sender_email or not sender_password:
        st.error("Email not configured.")
        return False
    subject = "🚨 High Risk Patient Alert"
    body = "ALERT: High Risk detected\n\n" + "\n".join([f"{k}: {v}" for k, v in patient_data.items()])
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = sender_email
    msg["To"] = receiver_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ---------------- APP ----------------
st.set_page_config(page_title="Patient Health Monitoring", layout="wide")
st.title("🏥 Patient Health Monitoring")

model = load_model()
ensure_users_csv()

# Sidebar
with st.sidebar:
    st.header("Account")
    menu = st.selectbox("Go to", ["Login", "Sign up", "About"])

# ---------- SIGNUP ----------
if menu == "Sign up":
    st.subheader("Create a new account")
    new_name = st.text_input("Full name")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    new_caretaker = st.text_input("Caretaker / Doctor email")
    if st.button("Create account"):
        if not (new_name and new_username and new_password and new_caretaker):
            st.warning("Please fill all fields.")
        else:
            users = read_users()
            if new_username in users["username"].values:
                st.error("Username already exists.")
            else:
                hashed = hash_password(new_password)
                ok, msg = add_user(new_username, new_name, hashed, new_caretaker)
                if ok:
                    st.success("Account created. Please login.")
                else:
                    st.error(msg)

# ---------- LOGIN ----------
elif menu == "Login":
    if not st.session_state.get("authenticated", False):
        st.subheader("Login to your account")
        login_username = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        if st.button("Login"):
            users = read_users()
            row = users.loc[users["username"] == login_username]
            if row.empty:
                st.error("User not found.")
            elif verify_password(row["password"].values[0], login_password):
                st.success(f"Welcome {row['name'].values[0]}!")
                st.session_state["user"] = login_username
                st.session_state["name"] = row["name"].values[0]
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")

    # Show vitals only if logged in
    if st.session_state.get("authenticated", False):
        username = st.session_state["user"]
        name = st.session_state.get("name", username)

        st.sidebar.write(f"Logged in as: {name}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

        # ---------------- VITALS INPUT ----------------
        st.subheader("Enter patient health details")
        heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 80)
        resp_rate = st.number_input("Respiratory Rate (breaths/min)", 8, 60, 16)
        body_temp = st.number_input("Body Temperature (°C)", 30.0, 45.0, 37.0)
        oxygen_sat = st.number_input("Oxygen Saturation (%)", 50, 100, 97)
        sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", 60, 200, 120)
        dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 140, 80)
        age = st.number_input("Age", 0, 120, 30)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        derived_bmi = st.number_input("Derived BMI", 10.0, 50.0, 22.0)
        derived_map = st.number_input("Derived MAP", 50.0, 140.0, 90.0)

        gender_encoded = 0 if gender == "Male" else 1
        features_array = np.array([[heart_rate, resp_rate, body_temp, oxygen_sat,
                                    sys_bp, dia_bp, age, gender_encoded, derived_bmi, derived_map]])

        patient_data = {
            "Heart Rate": heart_rate, "Respiratory Rate": resp_rate,
            "Body Temperature": body_temp, "Oxygen Saturation": oxygen_sat,
            "Systolic Blood Pressure": sys_bp, "Diastolic Blood Pressure": dia_bp,
            "Age": age, "Gender": gender, "Derived_BMI": derived_bmi, "Derived_MAP": derived_map
        }

        if st.button("Predict Risk"):
            pred = model.predict(features_array)[0]
            prob = float(np.max(model.predict_proba(features_array))) if hasattr(model, "predict_proba") else None
            risk_text = "High Risk" if pred == 1 else "Low Risk"

            normal_ranges = {
                "Heart Rate": (60, 100), "Respiratory Rate": (12, 20),
                "Body Temperature": (36.1, 37.2), "Oxygen Saturation": (95, 100),
                "Systolic Blood Pressure": (90, 120), "Diastolic Blood Pressure": (60, 80),
                "Derived_BMI": (18.5, 24.9), "Derived_MAP": (70, 100),
            }

            abnormal_vitals, precaution_msgs = [], []
            for vital, (low, high) in normal_ranges.items():
                value = float(patient_data[vital])
                if value < low or value > high:
                    abnormal_vitals.append(f"{vital}: {value}")
                    if vital in precautions:
                        precaution_msgs.append(f"{vital}: {precautions[vital]}")

            if pred == 1:
                st.error(f"⚠ {risk_text} (confidence={prob:.2f})")
            else:
                st.success(f"✅ {risk_text} (confidence={prob:.2f})")

            if abnormal_vitals:
                st.warning("Abnormal vitals detected: " + ", ".join(abnormal_vitals))
                if precaution_msgs:
                    st.info("Suggested precautions:\n- " + "\n- ".join(precaution_msgs))
            else:
                st.info("All vitals are within normal ranges.")

            if pred == 1:
                caretaker_email = read_users().loc[read_users()["username"] == username, "caretaker_email"].values[0]
                if send_email_alert(patient_data, caretaker_email):
                    st.info(f"📧 Email alert sent to {caretaker_email}")

            row = {"Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **patient_data, "Risk": risk_text}
            save_history(username, row)
            st.info("Saved prediction to your history.")

        # ---------------- HISTORY ----------------
        st.markdown("---")
        st.subheader("Your patient history")
        history_df = load_history(username)
        if history_df is not None and not history_df.empty:
            st.dataframe(history_df)

            st.markdown("### 📈 Summary of Vitals")
            summary_stats = history_df.describe().T[["mean", "min", "max"]].rename(
                columns={"mean": "Average", "min": "Minimum", "max": "Maximum"}).round(2)
            st.dataframe(summary_stats)

            history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=history_df, x="Date", y="Heart Rate", marker="o", label="Heart Rate", ax=ax)
            sns.lineplot(data=history_df, x="Date", y="Oxygen Saturation", marker="o", label="Oxygen Saturation", ax=ax)
            sns.lineplot(data=history_df, x="Date", y="Body Temperature", marker="o", label="Body Temp", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            sns.countplot(data=history_df, x="Risk", palette="coolwarm", ax=ax2)
            st.pyplot(fig2)

            high_count = (history_df["Risk"] == "High Risk").sum()
            low_count = (history_df["Risk"] == "Low Risk").sum()
            if high_count > low_count:
                st.error(f"⚠ Frequent high-risk readings ({high_count}). Medical review needed.")
            else:
                st.success(f"✅ Mostly stable readings ({low_count} low-risk cases).")

            st.download_button("📄 Download Full Patient Report (CSV)",
                               history_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"patient_report_{username}.csv")
            pdf_buffer = generate_pdf_report_line_by_line(history_df.tail(1), name, username)
            st.download_button("📄 Download Patient Report (PDF)", data=pdf_buffer,
                               file_name=f"report_{username}.pdf", mime="application/pdf")
        else:
            st.write("No history yet.")

# ---------- ABOUT ----------
elif menu == "About":
    st.write("""
    Patient Health Monitoring App  
    - Multi-user login with secure hashed passwords  
    - Per-user history stored locally  
    - Email alerts sent to caretaker (requires Gmail App Password)  
    - CSV = full patient history, PDF = latest entry  
    - Uses ML model: health_model.pkl  
    """)