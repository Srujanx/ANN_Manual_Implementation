# 🧠 Artificial Neural Network (ANN) From Scratch — Python + NumPy

### Author: Srujan  
📍 AI Undergraduate | Machine Learning • Deep Learning • Agentic AI  

---

## 🚀 Project Overview
This project implements an **Artificial Neural Network (ANN)** completely **from scratch**, without using any high-level ML frameworks such as TensorFlow, PyTorch, or scikit-learn.

Every step — from **forward propagation** to **backpropagation** and **gradient descent** — was built manually using only **NumPy**.  
The goal: to deeply understand **how a neural network actually learns** by calculating and verifying every update by hand.

---

## 🧩 Key Objectives
- Implement **feedforward and backpropagation** manually.  
- Derive **gradient descent** updates step-by-step (mathematically + in code).  
- Design and train a **two-layer neural network** for binary/multiclass classification.  
- Verify each computation manually using **pen-and-paper derivations**.  
- Build a working prototype that demonstrates convergence and learning dynamics.  

---

## ⚙️ Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3.x |
| Math / Computation | NumPy |
| Concepts | Linear Algebra, Matrix Calculus, Gradient Descent |
| ML Topics | Neural Networks, Activation Functions, Loss Optimization |

---

## 🧠 Architecture
Input Layer (n features)
↓
Hidden Layer 
↓
Output Layer (Sigmoid)


Each neuron updates through manually computed gradients.  
No optimizer shortcuts — only raw linear algebra and calculus.

---

## 🧮 Implementation Highlights
- **Forward Propagation:**  
  Computes activations layer by layer using dot products and non-linear activations.
- **Backward Propagation:**  
  Derives and applies partial derivatives of loss w.r.t. weights and biases.
- **Gradient Descent:**  
  Manual implementation controlling learning rate, convergence, and stopping criteria.
- **Evaluation:**  
  Accuracy tracking across epochs, verifying model learns expected patterns.

---

## 📊 Results
- Successful training convergence on small synthetic dataset.  
- Visualized **loss reduction** over epochs.  
- Verified gradient flow and parameter updates numerically.  

*(Note: focus of this project is correctness and clarity, not large-scale performance.)*

---

## 💡 Key Learnings
- True understanding of **how gradients flow** through the network.  
- Ability to debug learning failures at the matrix-operation level.  
- Appreciation for the complexity abstracted away by frameworks.  
- Stronger grasp of **ML fundamentals** before tackling advanced models (LLMs, CNNs, RNNs).  

---

## 📘 Next Steps
- Extend to **multi-layer deep networks**.  
- Add **momentum and adaptive learning** (e.g., Adam).  
- Implement **regularization** (dropout / L2).  
- Compare performance with **scikit-learn’s MLPClassifier** for validation.  
- Deploy a minimal **FastAPI microservice** to serve predictions.  

---

🧩 Folder Structure

- ANN_from_scratch.ipynb — Python implementation of ANN from scratch (no sklearn or keras)
- ann.pdf — Manual math derivations and handwritten calculations
- README.md — Project documentation (this file)

---

## 🔗 Connect
If you’re working on similar ML fundamentals or deploying educational AI projects, I’d love to connect or collaborate.

**LinkedIn:** https://linkedin.com/in/srujan77  
**GitHub:** https://github.com/Srujanx

---

### 🏁 Summary
This project isn’t about flashy frameworks — it’s about *understanding the math that makes ML work*.  
By implementing everything from scratch, I’ve built a foundation that scales naturally into **Deep Learning**.

---

**#MachineLearning #DeepLearning #ArtificialIntelligence #NumPy #Python #NeuralNetworks #GradientDescent #Backpropagation #AIEngineering**
