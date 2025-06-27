# DeepSentinel - AI Powered Security Surveillance System

![DeepSentinel Logo](docs/logo.png) _will be added later._

## 🛡️ Features
- Real-time weapon detection (guns, knives)
- Suspicious behavior analysis (fighting, loitering)
- Safety compliance monitoring (PPE detection)
- Multi-channel alert system (SMS, email, app)
- Voice-controlled interface
- Continuous learning through cloud feedback

---

## ⚙️ Installation

### Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
```

###  Install dependencies
```bash
pip install -r requirements.txt

# Start the application
python run.py
```

# Train custom model
python train.py --dataset data/processed --epochs 100
📂 Project Structure
```text
DeepSentinel/
├── config/              # Configuration files
├── data/                # Training datasets
├── deep_sentinel/       # Main application code
├── docs/                # Documentation
├── models/              # Pretrained and custom models
├── scripts/             # Utility scripts
├── tests/               # Tests
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── run.py               # Main entry point
└── train.py             # Training entry point
```

---

## 📘 Documentation

- [Workflow & System Design](docs/WORKFLOW.md)

---

## 🤝 Contributing
1. Fork the repository

2. Create your feature branch (git checkout -b feature/amazing-feature)

3. Commit your changes (git commit -m 'feat: add amazing feature')

4. Push to the branch (git push origin feature/amazing-feature)

5. Open a pull request

---

## 📜 License
Distributed under the [MIT License](./LICENSE). See the LICENSE file for more information.

---

_Thank you!_ 