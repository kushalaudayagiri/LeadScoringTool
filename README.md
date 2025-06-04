
# Lead Scoring Tool

This is a Streamlit web app for automated lead scoring using a Random Forest Classifier. The app automatically detects the target column from your uploaded CSV dataset, trains a model, evaluates it, and scores leads with categories (Low, Medium, High). You can also download the scored leads.

---

## Features

- Upload lead dataset as CSV
- Auto-detects target column for classification
- Preprocesses data with one-hot encoding
- Trains Random Forest Classifier on 70% of data
- Evaluates model performance (Accuracy, AUC)
- Scores leads with probability-based lead scoring
- Displays scored leads with color-coded categories
- Download scored leads as CSV file

---

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Recommended to use a virtual environment

### Installation

1. Clone the repository or download the script:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install the packages manually:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### Running the App

Run the Streamlit app with:

```bash
streamlit run app.py
```

Replace `app.py` with your script filename if different.

### Usage

- Upload a CSV file containing your leads data via the sidebar.
- The app will automatically detect the target column.
- Click **Train & Score Leads** to train the model and score your leads.
- View scored leads, model performance, and download results.

---

## Notes

- The target column should be categorical with a small number of unique values.
- The app uses a simple Random Forest model; further tuning can improve performance.
- Large datasets may take longer to process.

---

## License

MIT License

---

## Contact

For questions or feedback, contact Kushala Udayagiri at kushalaudayagiri@gmail.com
