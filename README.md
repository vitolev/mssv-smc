# Estimating Markov Switching Stochastic Volatility in Financial Markets using Sequential Monte Carlo Methods

This repository contains code for estimating Markov Switching Stochastic Volatility (MSSV) models in financial markets using Sequential Monte Carlo (SMC) methods, including Particle MCMC and SMC². The project is part of a Master’s thesis in Computer and Data Science by Vito Levstik.

---

## Table of Contents

- [Overview](#overview)  
- [Motivation](#motivation)  
- [Features](#features)  

---

## Overview

Financial market volatility often exhibits sudden shifts between low and high volatility regimes. MSSV models capture these dynamics by combining:

- **Latent stochastic volatility dynamics**  
- **Regime-switching behavior**  

Sequential Monte Carlo (SMC) methods allow efficient estimation of latent states and model parameters, handling models that are globally nonlinear and non-Gaussian.

---

## Motivation

- Understanding and forecasting volatility is crucial for:  
  - Risk management  
  - Portfolio optimization  
  - Derivatives pricing  

- Traditional methods (MLE, Kalman filter) fail for MSSV due to nonlinearity and regime-switching.  
- SMC, Particle MCMC, and SMC² provide a robust framework for inference.

---

## Features

- Generic **State-Space Model** base classes  
- **Parameter-less** and **state-less** model variants for testing  
- Implementation of **MSSV model**  
- Sequential Monte Carlo methods for latent state and parameter inference  
- Simulation-based validation framework  

---
