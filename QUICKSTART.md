# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Activate Virtual Environment
```powershell
cd s:\projects\Temp\Ecommerce_recommendation
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your PowerShell prompt.

### Step 2: Start Jupyter
```powershell
jupyter notebook
```

This opens your browser to Jupyter Lab.

### Step 3: Run First Notebook
1. Navigate to `notebooks/` folder
2. Open `01_data_loading.ipynb`
3. Press `Ctrl+A` then `Shift+Enter` to run all cells

**What happens:**
- ✅ MovieLens 100K dataset downloads (~5 MB)
- ✅ Data loads automatically
- ✅ Statistics and visualizations appear
- ✅ Files saved to `data/processed/`

### Step 4: Run Next Notebook
Once Step 3 completes:
1. Open `02_collaborative_filtering.ipynb`
2. Run all cells
3. You'll see model training and recommendations

### Step 5: Continue with Other Notebooks
- `03_matrix_factorization.ipynb` - SVD model
- `04_deep_learning.ipynb` - Neural network model

---

## 📊 What Each Notebook Does

| Notebook | Purpose | Time | Outputs |
|----------|---------|------|---------|
| 01_data_loading | Download & explore data | ~2 min | CSV files, plots |
| 02_collaborative_filtering | Train CF models | ~1 min | RMSE, recommendations |
| 03_matrix_factorization | Train SVD model | ~3 min | Latent factors, plots |
| 04_deep_learning | Train neural model | ~15 min | Trained model, plots |

---

## 🎯 Success Indicators

After each notebook, you should see:

**After 01_data_loading:**
```
MovieLens 100K Dataset Statistics
==================================================
Total Ratings: 100,000
Unique Users: 943
Unique Movies: 1,682
Rating Range: 1.0 - 5.0
Sparsity: 0.9356
```

**After 02_collaborative_filtering:**
```
User-Based CF Results:
RMSE: 0.95
MAE: 0.75

Top 10 recommendations for User 1:
[123, 456, 789, ...]
```

**After 03_matrix_factorization:**
```
SVD Model Results:
RMSE: 0.88
MAE: 0.68

Variance explained by 50 factors: 0.6234
```

**After 04_deep_learning:**
```
Neural Collaborative Filtering Results:
RMSE: 0.85
MAE: 0.65

Top 10 recommendations for User 1:
1. Movie 123 (Score: 4.85/5.0)
2. Movie 456 (Score: 4.72/5.0)
...
```

---

## 🔍 If Something Goes Wrong

### Error: "No module named 'tensorflow'"
```powershell
.\venv\Scripts\pip install tensorflow --upgrade
# Restart kernel: Kernel > Restart in Jupyter
```

### Error: Dataset not found
Run the first cell of 01_data_loading.ipynb - it auto-downloads.

### Error: "TruncatedSVD" not found
```powershell
.\venv\Scripts\pip install scikit-learn --upgrade
```

### Error: Kernel crashes during deep learning
The model is training! This is normal. Check output in terminal.

---

## 💻 System Requirements

- **RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** 1GB minimum
- **Python:** 3.9+ (current: 3.12)
- **Packages:** All listed in requirements.txt

---

## 📈 Expected Timeline

- **Total project time:** 30-40 minutes
  - Data loading: 2 min ✅
  - Collaborative filtering: 3 min ✅
  - Matrix factorization: 5 min ✅
  - Deep learning: 20-25 min ⏳
  - Evaluation & review: 5 min ✅

---

## 🎓 What You'll Learn

1. ✅ How to load and preprocess recommendation data
2. ✅ Implement collaborative filtering from scratch
3. ✅ Apply matrix factorization techniques
4. ✅ Build deep learning recommendation models
5. ✅ Evaluate models using proper metrics
6. ✅ Compare and rank different algorithms

---

## 🏁 After Completion

Your project structure will have:
- ✅ Raw dataset in `data/raw/`
- ✅ Processed data in `data/processed/`
- ✅ Trained models in `notebooks/`
- ✅ Comprehensive analysis and visualizations

**You're ready to:**
- Add to your GitHub portfolio
- Write a blog post about your experience
- Make it a resume project
- Deploy as an API
- Extend with more features

---

## 🚀 Ready?

1. Activate: `.\venv\Scripts\Activate.ps1`
2. Launch: `jupyter notebook`
3. Open: `notebooks/01_data_loading.ipynb`
4. Run: `Ctrl+A` then `Shift+Enter`

**The rest happens automatically!** 🎉
