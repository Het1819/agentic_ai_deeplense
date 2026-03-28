# DeepLense GSoC 2026 Evaluation Tests

This repository contains the completed evaluation tests for the **Google Summer of Code (GSoC) 2026** application for the **DeepLense** project, specifically targeting the **Agentic AI for Autonomous Gravitational Lensing Simulation Workflows** track.

## 👤 Applicant Information
* **Name:** Het
* **Target Project:** Agentic AI for Autonomous Gravitational Lensing Simulation Workflows
* **Organization:** ML4SCI (DeepLense)

---

## 📂 Repository Structure

The repository is divided into two main sections corresponding to the required GSoC tests:

### [1. Common Test I: Multi-Class Classification](./Test%201/)
A complete end-to-end deep learning pipeline built using PyTorch to classify simulated strong gravitational lensing images.
* **Architecture:** Custom Convolutional Neural Network (CNN).
* **Highlights:** Programmatic 90:10 stratified data split, controlled data augmentation, and comprehensive evaluation using One-vs-Rest ROC curves and AUC scores.
* **Key Files:** `main_project.ipynb`, `lens_classifier_final.pth`

### [2. Specific Test II: Agentic AI](./Test%202%20(Agentic%20AI)/)
A production-grade, interactive scientific assistant built with Pydantic AI that translates natural language prompts into validated simulation configurations for the `DeepLenseSim` pipeline.
* **Architecture:** LLM-driven autonomous agent utilizing strictly typed Pydantic models for tool execution.
* **Highlights:** * Supports multiple published configurations (`model_i`, `model_ii`).
  * Implements a robust **Human-in-the-Loop** checkpoint using a SHA-256 cryptographic digest to prevent unauthorized or hallucinated simulation executions.
  * API-key-free reproducibility for evaluators via a mock `FunctionModel`.
* **Key Files:** `agent_demo.ipynb`, `deeplense_agent/` module.

---

## 🚀 Getting Started

To explore the solutions, please navigate to the respective test directories. Each folder contains its own detailed `README.md` with specific installation instructions, strategy discussions, and execution steps.

**General Requirements:**
* Python 3.11+
* `torch`, `torchvision`, `scikit-learn` (Test I)
* `pydantic-ai-slim`, `lenstronomy`, `astropy` (Test II)