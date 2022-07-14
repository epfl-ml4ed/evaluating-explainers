# evaluating-explainers

This repository is the official implementation of the EDM 2022 paper entitled "Evaluating the Explainers: A Comparison of Explainability Techniques for Black-Box Success Prediction in MOOCs" written by [Vinitra Swamy](http://github.com/vinitra), [Bahar Radmehr](https://github.com/BaharRadmehr), [Natasa Krco](https://github.com/Nat998), [Mirko Marras](https://www.mirkomarras.com/), and [Tanja Käser](https://people.epfl.ch/tanja.kaeser/?lang=en). 

Experiments are located in `scripts/`, corresponding directly to the experimental methodology mentioned in the paper. 

> **July 1, 2022: This version of the evaluating-explainers code is not fully cleaned and is subject to refactoring in the next months.**

## Project overview

We implement five state-of-the-art methodologies for explaining black-box machine learning models (LIME, PermutationSHAP, KernelSHAP, DiCE, CEM) on the downstream task of student performance prediction for five massive open online courses. Our experiments demonstrate that the families of explainers **do not agree** with each other on feature importance for the same Bidirectional LSTM models with the same representative set of students. 

We use Principal Component Analysis, Jensen-Shannon distance, and Spearman's rank-order correlation to quantitatively cross-examine explanations across methods and courses. Our results come to the concerning conclusion that the choice of explainer is an important decision and is in fact paramount to the interpretation of the predictive results, even more so than the course the model is trained on. This project started in the ML4ED laboratory at EPFL in May 2021, and will be featured at EDM 2022 at Durham University. 

## Usage guide

0. Install relevant dependencies with `pip install -r requirements.txt`.

1. Extract relevant features sets (`BouroujeniEtAl`, `MarrasEtAl`, `LalleConati`, and `ChenCui`) through the ML4ED lab's EDM 2021 contribution on [benchmarks for feature predictive power](https://github.com/epfl-ml4ed/flipped-classroom). Place the results of these feature extraction scripts in `data/`. A toy course example of extracted features is included in the `data/` folder.

2. Use the trained BiLSTM model for each course in the `models/` folder. Alternatively, train the model yourself with `python scripts/LSTM.py`, based on the BO (Behavior-Only) baseline models from the ML4ED lab's L@S 2022 contribution on [meta transfer learning](https://github.com/epfl-ml4ed/meta-transfer-learning).

3. Select your uniformly sampled subset of points (for time-efficiency, reducing the experiments to hours instead of weeks) by running `python scripts/uniform.py`.

4. Run your desired experiment from `scripts/` by executing the explainability script with Python 3.7 or higher (i.e. `python scripts/LIME.py`).

## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know. Feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations
If you find this code useful in your work, please cite our paper:

```
Swamy, V., Radmehr, B., Krco, N., Marras, M., Käser, T. (2022). 
Evaluating the Explainers: A Comparison of Explainability Techniques for Black-Box Success Prediction in MOOCs. 
In: Proceedings of the 15th International Conference on Educational Data Mining (EDM 2022).
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the [MIT License](LICENSE).

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the [MIT License](LICENSE) for details.
