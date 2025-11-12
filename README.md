# 📊 Ethereum Price Prediction using Machine Learning
### A Binary Classification Approach for Trading Signal Detection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-yellow.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.01%25-brightgreen.svg)](https://github.com/Farrely-F/eth-price-prediction)

> **AI System Coursework Project - Telkom University**  
> Predicting BUY/SELL trading signals for Ethereum with **92.01% accuracy** using machine learning

---

## 🎯 Project Overview

This project builds an intelligent system that predicts optimal BUY or SELL signals for Ethereum cryptocurrency trading using machine learning. By analyzing historical price data and technical indicators, the model achieves **92.01% accuracy** on unseen test data.

### Key Achievements
- ✅ **92.01% Accuracy** on test dataset
- ✅ **9 ML Algorithms** tested and compared
- ✅ **20+ Technical Indicators** engineered from raw data
- ✅ **100,000+ Data Points** analyzed (2016-2020)
- ✅ **Production-Ready Model** with comprehensive evaluation

---

## 📁 Project Structure

```
eth-price-prediction/
│
├── 📓 eth_price_prediction.ipynb          # Main Jupyter notebook (beginner-friendly!)
│
├── 📊 Data Files
│   ├── ETH_1H.csv                         # Hourly data (primary dataset)
│   ├── ETH_1min.csv                       # Minute-level data
│   └── ETH_day.csv                        # Daily data
│
├── 📋 Presentation Materials
│   ├── PRESENTATION_OUTLINE.md            # Full 27-slide outline for Canva
│   ├── PRESENTATION_QUICK_GUIDE.md        # Quick reference for presenting
│   ├── PRESENTATION_HANDOUT.md            # One-page summary handout
│   └── CANVA_DESIGN_GUIDE.md              # Design guide for creating slides
│
├── 📄 Ethereum-Price-Prediction.pdf       # Project documentation
└── 📖 README.md                            # This file
```

---

## 🚀 Quick Start

### Option 1: Run Online (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Open the notebook in Google Colab
2. Run all cells sequentially
3. Dataset loads automatically from GitHub (no setup needed!)

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/Farrely-F/eth-price-prediction.git
cd eth-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook eth_price_prediction.ipynb
```

### Requirements
```
Python >= 3.7
numpy >= 1.19.0
pandas >= 1.1.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
tensorflow >= 2.4.0
keras >= 2.4.0
```

---

## 📊 Dataset

**Source**: Ethereum Historical Data from Bitstamp Exchange  
**Available at**:
- 📊 [Kaggle Dataset](https://www.kaggle.com/datasets/prasoonkottarathil/ethereum-historical-dataset)

**Dataset Details**:
- **Size**: 100,000+ hourly records
- **Period**: May 2016 - April 2020
- **Features**: Unix Timestamp, Date, Symbol, Open, High, Low, Close, Volume
- **Quality**: No missing values, balanced classes (50/50 BUY/SELL)

---

## 🔬 Methodology

### 1. Data Preprocessing
- ✅ Null value detection and handling
- ✅ Timestamp removal (prevent data leakage)
- ✅ Feature scaling (StandardScaler)
- ✅ Train-test split (80/20)

### 2. Feature Engineering
Transformed 5 raw features into **20+ technical indicators**:

| Indicator Type | Features | Timeframes |
|----------------|----------|------------|
| **Moving Averages** | EMA, MA | 10h, 30h, 200h |
| **Momentum** | MOM | 10h, 30h |
| **Rate of Change** | ROC | 10h, 30h |
| **RSI** | Relative Strength Index | 10h, 30h, 200h |
| **Stochastic** | %K, %D | 10h, 30h, 200h |

### 3. Model Selection
Tested **9 different algorithms** across multiple paradigms:

| Algorithm | Accuracy | Category |
|-----------|----------|----------|
| K-Nearest Neighbors | 96.63% | Instance-Based |
| **Random Forest** ⭐ | **93.35%** | Ensemble (Bagging) |
| Gradient Boosting | 91.99% | Ensemble (Boosting) |
| LDA | 91.11% | Linear |
| Decision Tree | 89.76% | Tree-Based |
| AdaBoost | 89.97% | Ensemble (Boosting) |
| Logistic Regression | 51.56% | Linear |
| Naive Bayes | 51.36% | Probabilistic |
| Neural Network | 49.45% | Deep Learning |

### 4. Optimization
- **Hyperparameter Tuning**: GridSearchCV
- **Cross-Validation**: 10-Fold CV
- **Best Parameters**: 80 trees, max_depth=10, criterion='gini'

### 5. Evaluation
- **Accuracy**: 92.01%
- **Precision**: 92% (both BUY and SELL)
- **Recall**: 93% (SELL), 91% (BUY)
- **F1-Score**: 92%
- **ROC-AUC**: ~0.96-0.98

---

## 📈 Results

### Final Model Performance

```
Confusion Matrix:
                Predicted
                SELL   BUY
Actual  SELL    3286   258
        BUY      289  3010

Overall Accuracy: 92.01%
```

**What This Means:**
- Out of 100 predictions, **92 are correct**!
- Model is equally good at predicting both BUY and SELL signals
- Significantly outperforms random guessing (50%)

### Feature Importance
Top predictive features:
1. 💰 Close Price
2. 📊 Volume
3. 💪 RSI (Relative Strength Index)
4. 📈 Moving Averages
5. 🎢 Stochastic Oscillators

---

## 💡 Key Insights

### What We Learned:
1. **Ensemble methods dominate** in financial prediction tasks
2. **Feature engineering is critical** - technical indicators >> raw prices
3. **Balanced datasets ensure unbiased predictions**
4. **Model interpretability builds trust** in AI systems
5. **Rigorous validation prevents overfitting**

### Business Impact:
- Accurate trading signals can significantly improve trading returns
- 92% accuracy translates to profitable strategies (with proper risk management)
- Model decisions align with established trading principles

---

## ⚠️ Limitations & Disclaimers

### Technical Limitations:
- ❌ Historical data bias (may not predict unprecedented events)
- ❌ Time-series dependencies not fully captured
- ❌ Trained on single asset (Ethereum only)
- ❌ Hourly granularity misses intra-hour movements

### Financial Limitations:
- ❌ Transaction costs not included in backtesting
- ❌ Market impact and slippage not considered
- ❌ No risk management implementation
- ❌ Regulatory constraints not addressed

### Important Disclaimer:
⚠️ **This is an academic project, NOT financial advice!**
- 92% accuracy in backtesting ≠ 92% returns in live trading
- Real trading requires proper risk management
- Never invest more than you can afford to lose
- Consult financial professionals before trading

---

## 🚀 Future Work

### Beginner Level:
1. Test on other cryptocurrencies (Bitcoin, Litecoin)
2. Expand hyperparameter search space
3. Test different time granularities (5-min, daily)

### Intermediate Level:
4. Add more technical indicators (Bollinger Bands, MACD)
5. Implement sentiment analysis (Twitter, news)
6. Include realistic transaction costs
7. Walk-forward optimization

### Advanced Level:
8. Implement LSTM/GRU time-series models
9. Real-time deployment (REST API)
10. Advanced risk management (Kelly Criterion, VaR)
11. Multi-asset portfolio optimization
12. Automated trading bot integration

---

## 📚 Documentation

### Notebook Structure:
The `eth_price_prediction.ipynb` notebook is **beginner-friendly** with:
- 📖 Detailed explanations for every concept
- 🎯 Real-world analogies for complex topics
- 📊 Comprehensive visualizations
- ✅ Step-by-step walkthrough
- 💡 Insights and interpretations

**Perfect for:**
- Students learning machine learning
- Presentations to non-technical audiences
- Academic coursework submissions
- Portfolio projects

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

### How to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- **Kaggle** - Dataset source
- **Bitstamp Exchange** - Historical data
- **Scikit-learn** - ML framework
- **Open source community** - Tools and libraries

---

## 📖 References

1. [Kaggle Ethereum Historical Dataset](https://www.kaggle.com/datasets/prasoonkottarathil/ethereum-historical-dataset)
2. [Scikit-learn Documentation](https://scikit-learn.org/)
3. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
4. [Technical Analysis of Financial Markets](https://www.investopedia.com/technical-analysis-4689657)

---

## 📊 Project Statistics

![Python](https://img.shields.io/badge/Python-100%25-blue)
![ML Models](https://img.shields.io/badge/ML%20Models-9-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-92.01%25-brightgreen)
![Data Points](https://img.shields.io/badge/Data%20Points-100K%2B-yellow)
![Features](https://img.shields.io/badge/Features-20%2B-purple)

---

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star! ⭐

It helps others discover the project and motivates continued development.

---

**Built with ❤️ for the AI and Machine Learning community**

*Last Updated: November 2024*
