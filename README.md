# Automated Dataset Quality Scoring and Fairness Auditing System

**A complete backend system for dataset quality analysis, fairness auditing, AI explanations, authentication, and persistence.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)]()

---

## 🎓 Team Members

- **Atharva Dange** (1032221013) – Backend Owner / API & Core Engine
- **Ritika Palai** (1032221042)
- **Raeva Dashputre** (1032221426)
- **Khushi Bhangdia** (1032220317)

**Institution:** MIT WPU Pune  
**Panel:** CSBS-A  

---

## 🚀 Project Overview

This system provides a complete REST API for:

- **Dataset Upload & Validation**
- **Data Quality Scoring**
- **Fairness Auditing**
- **AI-Based Explanations**
- **SQLite Persistence**
- **User Authentication**

The system helps evaluate whether datasets are **clean, reliable, and fair for machine learning use cases**.

---

## ✨ Features

✅ Dataset Upload API  
✅ Data Quality Scoring (completeness, validity, consistency, uniqueness)  
✅ Fairness Auditing (bias detection across protected attributes)  
✅ AI Explanation Engine  
✅ SQLite Persistence  
✅ Restart-safe report retrieval  
✅ Session-based Authentication  
✅ REST API ready for frontend integration  

---

## 🏗️ Architecture

```text
Frontend (Teammates)
        │
        ▼
   Flask REST API
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
Auth   Quality       Fairness
DB     Scoring       Audit
 │         │             │
 └─────────┴──────┬──────┘
                  ▼
           AI Explanation
                  ▼
              SQLite DB
```

---

## 📦 Tech Stack

- **Language:** Python 3.9+
- **Framework:** Flask
- **Database:** SQLite + SQLAlchemy
- **Data Processing:** Pandas, NumPy
- **Authentication:** Flask Session
- **API Testing:** Postman

---

## 🔧 Setup

### Clone
```bash
git clone <your-repo-url>
cd fairness-detection-system
```

### Create virtual environment
```bash
python -m venv venv
```

### Activate
**Windows**
```bash
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run server
```bash
python run.py
```

---

## 🌐 API Endpoints

### Authentication
```text
POST /auth/register
POST /auth/login
POST /auth/logout
GET  /auth/me
```

### Dataset APIs
```text
POST /api/upload
GET  /api/quality/<dataset_id>
POST /api/audit
POST /api/explain
GET  /api/results/<dataset_id>
GET  /api/datasets
```

---

## 📁 Project Structure

```text
backend/
├── api/
├── auth/
├── data_processing/
├── database/
├── explainer/
├── fairness/
├── quality/

data/
docs/
app.db
run.py
requirements.txt
README.md
```

---

## 📊 Current Status

```text
✅ Backend Complete
✅ Authentication Complete
✅ Database Persistence Complete
✅ AI Explanation Complete
⏳ Frontend Integration Pending
```

---

## 🤝 Frontend Handoff

Backend is fully ready for teammate frontend integration.

Teammates can directly connect using the provided REST endpoints.

---

## 👨‍💻 Author

**Atharva Dange**  
MIT WPU Pune  
GitHub: https://github.com/Danthr

