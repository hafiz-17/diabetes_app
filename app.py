"""DiabetesAI Flask Application v4 – separate predict / result pages + session"""

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import joblib, os, io, datetime, json
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__)
app.secret_key = "diabetesai-secret-key-2025"

BASE  = os.path.dirname(os.path.abspath(__file__))
MDIR  = os.path.join(BASE, "models")
model      = joblib.load(os.path.join(MDIR, "diabetes_model.pkl"))
le_gender  = joblib.load(os.path.join(MDIR, "le_gender.pkl"))
le_smoking = joblib.load(os.path.join(MDIR, "le_smoking.pkl"))
FEATURES   = joblib.load(os.path.join(MDIR, "features.pkl"))

FEATURE_NICE = {
    "blood_glucose_level": "Blood Glucose",
    "HbA1c_level":         "HbA1c Level",
    "bmi":                 "BMI",
    "age":                 "Age",
    "hypertension":        "Hypertension",
    "heart_disease":       "Heart Disease",
    "smoking_history":     "Smoking History",
    "gender":              "Gender",
}
CLINICAL_NOTES = {
    "Blood Glucose":  "Direct glucose measurement – primary diagnostic value",
    "HbA1c Level":    "3-month average glucose control indicator",
    "BMI":            "Obesity strongly correlates with insulin resistance",
    "Age":            "Risk increases significantly after age 45",
    "Hypertension":   "Metabolic syndrome component, co-occurs with diabetes",
    "Heart Disease":  "Shared cardiovascular-metabolic risk pathway",
    "Smoking History":"Impairs insulin sensitivity and pancreatic function",
    "Gender":         "Biological sex influences metabolic risk profile",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_input_df(d):
    gender_enc  = le_gender.transform([d["gender"]])[0]
    smoking_enc = le_smoking.transform([d["smoking_history"]])[0]
    row = {
        "gender":              gender_enc,
        "age":                 float(d["age"]),
        "hypertension":        int(d["hypertension"]),
        "heart_disease":       int(d["heart_disease"]),
        "smoking_history":     smoking_enc,
        "bmi":                 float(d["bmi"]),
        "HbA1c_level":         float(d["HbA1c_level"]),
        "blood_glucose_level": float(d["blood_glucose_level"]),
    }
    return pd.DataFrame([row], columns=FEATURES)

def classify_risk(prob_frac):
    p = prob_frac * 100
    display = max(round(p, 1), 0.1) if p > 0 else 0.0
    if p < 30:   return "Low Risk",    "low",    "#10b981", display
    elif p < 60: return "Medium Risk", "medium", "#f59e0b", display
    else:        return "High Risk",   "high",   "#ef4444", display

def get_clinical_status(field, value):
    ranges = {
        "blood_glucose_level": [(70,99,"Normal","#10b981"),(100,125,"Pre-diabetic","#f59e0b"),(126,600,"Diabetic","#ef4444")],
        "HbA1c_level": [(4.0,5.6,"Normal","#10b981"),(5.7,6.4,"Pre-diabetic","#f59e0b"),(6.5,15,"Diabetic","#ef4444")],
        "bmi": [(10,18.4,"Underweight","#3b82f6"),(18.5,24.9,"Normal","#10b981"),(25,29.9,"Overweight","#f59e0b"),(30,70,"Obese","#ef4444")],
    }
    if field not in ranges: return "N/A", "#888"
    v = float(value)
    for lo, hi, label, clr in ranges[field]:
        if lo <= v <= hi: return label, clr
    return "Out of range", "#888"

def build_radar(d):
    def norm(val, lo, hi):
        return round(min(max((float(val)-lo)/(hi-lo)*100, 0), 100), 1)
    patient = [
        norm(d.get("blood_glucose_level",85), 70, 300),
        norm(d.get("HbA1c_level",5), 4, 12),
        norm(d.get("bmi",22), 15, 50),
        norm(d.get("age",40), 0, 100),
        int(d.get("hypertension",0))*100,
        int(d.get("heart_disease",0))*100,
    ]
    return {"patient": patient, "healthy":[20,15,25,40,0,0],
            "labels":["Blood Glucose","HbA1c","BMI","Age","Hypertension","Heart Disease"]}

def get_advice(risk_label, d):
    base = {
        "Low Risk": [
            "Maintain regular physical activity — aim for 150 min/week of moderate exercise.",
            "Continue a balanced diet rich in vegetables, whole grains, and lean proteins.",
            "Schedule annual check-ups to monitor blood glucose and HbA1c.",
            "Stay hydrated and aim for 7–9 hours of quality sleep each night.",
            "Keep stress low through mindfulness or leisure activities.",
        ],
        "Medium Risk": [
            "Reduce sugar, refined carbohydrates, and ultra-processed foods immediately.",
            "Increase physical activity to at least 150–300 min/week.",
            "Monitor fasting blood glucose regularly with a home glucometer.",
            "Consult your doctor for a comprehensive metabolic panel and HbA1c test.",
            "Work with a registered dietitian for a personalised meal plan.",
        ],
        "High Risk": [
            "Consult an endocrinologist as soon as possible — do not delay.",
            "Begin strict blood glucose monitoring: fasting, post-meal, and bedtime.",
            "Follow a low glycaemic index diet strictly under medical supervision.",
            "Discuss medication or insulin therapy options with your physician.",
            "Start a medically supervised exercise programme.",
            "Schedule HbA1c testing every 3 months without exception.",
        ],
    }[risk_label]
    if float(d.get("bmi",0)) > 30:
        base.append("Your BMI indicates obesity — structured weight management is critical.")
    if float(d.get("blood_glucose_level",0)) > 200:
        base.append("Blood glucose critically elevated — seek immediate medical attention.")
    if float(d.get("HbA1c_level",0)) >= 6.5:
        base.append("HbA1c ≥ 6.5% meets the clinical diagnostic threshold for diabetes.")
    if int(d.get("hypertension",0)):
        base.append("Hypertension compounds diabetes risk — monitor blood pressure daily.")
    if d.get("smoking_history") == "current":
        base.append("Active smoking significantly impairs insulin sensitivity — cessation is urgent.")
    return base

def validate_payload(d):
    required = {"gender":"Gender","age":"Age","bmi":"BMI",
                "blood_glucose_level":"Blood Glucose Level","HbA1c_level":"HbA1c Level",
                "hypertension":"Hypertension","heart_disease":"Heart Disease",
                "smoking_history":"Smoking History"}
    for key, label in required.items():
        if str(d.get(key,"")).strip() == "":
            return f"'{label}' is required."
    checks = [("age",1,120,"Age must be 1–120"),("bmi",10,70,"BMI must be 10–70 kg/m²"),
              ("blood_glucose_level",50,600,"Blood Glucose must be 50–600 mg/dL"),
              ("HbA1c_level",3,15,"HbA1c must be 3–15%")]
    for key, lo, hi, msg in checks:
        try:
            v = float(d[key])
            if not (lo <= v <= hi): return msg
        except: return f"'{key}' must be a valid number."
    if d["gender"] not in list(le_gender.classes_): return "Invalid gender value."
    if d["smoking_history"] not in list(le_smoking.classes_): return "Invalid smoking history."
    return None

def run_prediction(d):
    X = build_input_df(d)
    prob = float(model.predict_proba(X)[0][1])
    label, risk_key, color, pct = classify_risk(prob)
    feat_impact = sorted([
        {"feature": FEATURE_NICE[f], "key": f, "score": round(float(v)*100, 2)}
        for f,v in zip(FEATURES, model.feature_importances_)
    ], key=lambda x: -x["score"])
    clinical = {}
    for fld in ["blood_glucose_level","HbA1c_level","bmi"]:
        s,c = get_clinical_status(fld, d.get(fld,0))
        clinical[fld] = {"status":s,"color":c,"value":d.get(fld)}
    return {
        "probability": pct, "risk_label": label, "risk_key": risk_key,
        "risk_color": color, "feat_impact": feat_impact, "clinical": clinical,
        "radar": build_radar(d), "advice": get_advice(label, d), "input": d,
    }

# ── Page routes ────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET"])
def predict_page():
    smoking_opts = [{"val": c, "label": c.replace("No Info","No Information").replace("not current","Not Current").title()}
                    for c in le_smoking.classes_]
    error = session.pop("predict_error", None)
    return render_template("predict.html", smoking_opts=smoking_opts, error=error)

@app.route("/predict", methods=["POST"])
def predict_submit():
    """Form POST → validate → run model → store in session → redirect to result."""
    d = {
        "name":                request.form.get("name","").strip() or "Patient",
        "gender":              request.form.get("gender",""),
        "age":                 request.form.get("age",""),
        "bmi":                 request.form.get("bmi",""),
        "blood_glucose_level": request.form.get("blood_glucose_level",""),
        "HbA1c_level":         request.form.get("HbA1c_level",""),
        "hypertension":        request.form.get("hypertension",""),
        "heart_disease":       request.form.get("heart_disease",""),
        "smoking_history":     request.form.get("smoking_history",""),
    }
    err = validate_payload(d)
    if err:
        session["predict_error"] = err
        session["predict_form"]  = d
        return redirect(url_for("predict_page"))
    try:
        result = run_prediction(d)
        session["last_result"] = json.dumps(result)
        return redirect(url_for("result_page"))
    except Exception as e:
        session["predict_error"] = f"Prediction failed: {e}"
        session["predict_form"]  = d
        return redirect(url_for("predict_page"))

@app.route("/result")
def result_page():
    raw = session.get("last_result")
    if not raw:
        return redirect(url_for("predict_page"))
    result = json.loads(raw)
    return render_template("result.html", result=result)

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/about")
def about():
    importances = {FEATURE_NICE[f]: round(float(v)*100,2) for f,v in zip(FEATURES, model.feature_importances_)}
    importances_sorted = sorted(importances.items(), key=lambda x: -x[1])
    return render_template("about.html", importances=importances_sorted)

# ── API routes (for what-if and PDF) ──────────────────────────────────────────
@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    try:
        d = request.json or {}
        err = validate_payload(d)
        if err: return jsonify({"success":False,"error":err}), 400
        X = build_input_df(d)
        prob = float(model.predict_proba(X)[0][1])
        label, risk_key, _, pct = classify_risk(prob)
        return jsonify({"success":True,"probability":pct,"risk_label":label,"risk_key":risk_key})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 400

@app.route("/api/report", methods=["POST"])
def api_report():
    data = request.json
    prob = float(data["probability"]); risk_label = data["risk_label"]
    inp = data["input"]; advice = data["advice"]
    feat_impact = data["feat_impact"]; clinical = data.get("clinical",{})
    now = datetime.datetime.now()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2.2*cm)

    BLUE = colors.HexColor("#3b82f6"); GREEN = colors.HexColor("#10b981")
    ORG  = colors.HexColor("#f59e0b"); RED   = colors.HexColor("#ef4444")
    DARK = colors.HexColor("#0f172a")
    rc = {"Low Risk":GREEN,"Medium Risk":ORG,"High Risk":RED}.get(risk_label, BLUE)

    def sty(**kw): return ParagraphStyle("x",**kw)
    sNorm = sty(fontSize=9, leading=14, fontName="Helvetica")
    sBold = sty(fontSize=9, leading=14, fontName="Helvetica-Bold")
    sHead = sty(fontSize=13, leading=16, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=12, spaceAfter=6)
    sTitl = sty(fontSize=22, leading=26, fontName="Helvetica-Bold", textColor=BLUE, alignment=TA_CENTER)
    sSub  = sty(fontSize=10, leading=14, fontName="Helvetica", textColor=colors.grey, alignment=TA_CENTER)
    sDisc = sty(fontSize=7.5, leading=11, fontName="Helvetica-Oblique", textColor=colors.grey, alignment=TA_CENTER)

    E = []
    E.append(Paragraph("🏥  DiabetesAI — Medical Health Report", sTitl))
    E.append(Paragraph("AI-Powered Diabetes Risk Assessment", sSub))
    E.append(Paragraph(f"Generated: {now.strftime('%A, %B %d %Y  •  %I:%M %p')}  |  ID: DPS-{now.strftime('%Y%m%d%H%M%S')}", sSub))
    E.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceBefore=8, spaceAfter=12))

    E.append(Paragraph("Patient Information", sHead))
    pdata = [
        [Paragraph("Field",sBold),Paragraph("Value",sBold),Paragraph("Field",sBold),Paragraph("Value",sBold)],
        ["Full Name",inp.get("name","N/A"),"Gender",inp.get("gender","–")],
        ["Age",f"{inp.get('age','–')} years","BMI",f"{inp.get('bmi','–')} kg/m²"],
        ["Hypertension","Yes" if int(inp.get("hypertension",0)) else "No","Heart Disease","Yes" if int(inp.get("heart_disease",0)) else "No"],
        ["Smoking Hx",str(inp.get("smoking_history","–")).capitalize(),"HbA1c",f"{inp.get('HbA1c_level','–')} %"],
        ["Blood Glucose",f"{inp.get('blood_glucose_level','–')} mg/dL","Exam Date",now.strftime("%d %B %Y")],
    ]
    pt = Table(pdata, colWidths=[3.5*cm,5.5*cm,3.5*cm,5.5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),BLUE),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",(2,1),(2,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
        ("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),("PADDING",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    E.append(pt)

    E.append(Spacer(1,10)); E.append(Paragraph("Prediction Result", sHead))
    cond = "Diabetic" if prob >= 50 else "Non-Diabetic"
    desc = {"Low Risk":"Low probability. Continue healthy habits.","Medium Risk":"Moderate risk. Lifestyle intervention recommended.","High Risk":"High probability. Urgent specialist consultation advised."}[risk_label]
    rdata = [["Risk Level","Probability","Condition","Summary"],[risk_label,f"{prob:.1f}%",cond,desc]]
    rt = Table(rdata, colWidths=[3*cm,3*cm,3*cm,9*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),DARK),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("BACKGROUND",(0,1),(-1,1),rc),("TEXTCOLOR",(0,1),(-1,1),colors.white),("FONTNAME",(0,1),(-1,1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("PADDING",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),1,colors.white),
    ]))
    E.append(rt)

    E.append(Spacer(1,8)); E.append(Paragraph("Clinical Biomarker Analysis", sHead))
    clin_rows = [[Paragraph(h,sBold) for h in ["Biomarker","Your Value","Status","Normal Range","Clinical Note"]],
        ["Blood Glucose",f"{inp.get('blood_glucose_level','–')} mg/dL",clinical.get("blood_glucose_level",{}).get("status","–"),"70–99 mg/dL fasting","Primary diagnostic indicator"],
        ["HbA1c",f"{inp.get('HbA1c_level','–')} %",clinical.get("HbA1c_level",{}).get("status","–"),"< 5.7% Normal","3-month average glucose"],
        ["BMI",f"{inp.get('bmi','–')} kg/m²",clinical.get("bmi",{}).get("status","–"),"18.5–24.9 Healthy","Predictor of insulin resistance"],
    ]
    ct = Table(clin_rows, colWidths=[3*cm,2.8*cm,2.6*cm,3.8*cm,5.8*cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),BLUE),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
        ("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),("PADDING",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    E.append(ct)

    E.append(Spacer(1,8)); E.append(Paragraph("Top Risk Factors", sHead))
    fi_rows = [[Paragraph(h,sBold) for h in ["Factor","Importance","Impact","Clinical Note"]]]
    for fi in feat_impact[:6]:
        sc = fi["score"]; lvl = "High" if sc>15 else "Medium" if sc>7 else "Low"
        fi_rows.append([fi["feature"],f"{sc:.2f}%",lvl,CLINICAL_NOTES.get(fi["feature"],"–")])
    fit = Table(fi_rows, colWidths=[3.5*cm,3*cm,2.5*cm,9*cm])
    fit.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),BLUE),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
        ("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),("PADDING",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    E.append(fit)

    E.append(Spacer(1,8)); E.append(Paragraph("Health Recommendations", sHead))
    for adv in advice:
        E.append(Paragraph(adv, sNorm)); E.append(Spacer(1,3))

    E.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceBefore=12))
    E.append(Paragraph("⚕️ DISCLAIMER: This report is generated by an AI for informational purposes only and does NOT constitute medical advice. Always consult a qualified healthcare professional.", sDisc))

    doc.build(E); buf.seek(0)
    safe = inp.get("name","Patient").replace(" ","_")
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=f"DiabetesAI_{safe}_{now.strftime('%Y%m%d')}.pdf")

@app.errorhandler(404)
def not_found(e): return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
