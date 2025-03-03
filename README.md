# Data-Driven Decision Making for Sustainable Abalone Harvesting at OceanHarvest Inc.
Simon Hirlaender -  A course for companies
## Project Overview

This project explores how to make better decisions using data science and artificial intelligence (AI) in the context of sustainable ocean harvesting, focusing on **OceanHarvest Inc.**, a company specializing in abalone diving. The core challenge is to help OceanHarvest maximize its profits while strictly adhering to environmental regulations that protect abalone populations.

This involves:

*   **Accurately identifying valuable male abalone.**
*   **Minimizing costly errors (collecting females or young abalone, which results in penalties).**
* **Managing the inventory in an optimal way, to minimize the costs.**
* **Making the most of the diver's time and energy.**
* **Making good decisions even if we don't know the future.**
* **Taking into account the uncertainties.**

To tackle these challenges, we use a combination of data analysis, statistical modeling, machine learning, forecasting, and advanced decision-making techniques.

## Key Techniques

We combine several powerful techniques:

1.  **Exploratory Data Analysis (EDA):** Understanding the data and the relationship between each feature.
2.  **Statistical Modeling:** Using skewed normal distributions to model the age of abalone and make predictions.
3.  **Machine Learning:** Using neural networks to improve the prediction.
4.  **Profit-Based Decision Making:** Tailoring our models to maximize profit and minimize the costs.
5. **Generative Model:** Creating a model to generate new data, similar to the original data.
6.  **Bayesian Forecasting:** Using Bayesian Structural Time Series (BSTS) models with the PyMC library to predict future demand for abalone.
7.  **Dynamic Programming:** Using Dynamic Programming to optimize the inventory and the diver's actions, using methods like Value Iteration and the Wagner-Whitin algorithm.
8.  **Markov Decision Processes (MDPs):** Modeling the diver's decision-making process as an MDP to find the best strategy for maximizing profit during a dive.

## Notebook Structure

This project is presented in two main notebooks:

1.  **`Decision_theory.ipynb`:**

    *   **Exploratory Data Analysis (EDA):** We start by exploring the abalone dataset. We look at the distributions of different characteristics (like the number of rings) and visualize the relationships between features using plots and heatmaps.
    *   **Statistical Model:** We build a simple statistical model using skewed normal distributions to predict the sex of the abalone based on the number of rings. This model helps us understand the link between the age and the sex.
    * **Generative Model:** We then create a simple generative model, that is able to produce new data, similar to the original data.
    *   **Threshold Optimization:** We find the best decision threshold for our statistical model to maximize profit while minimizing penalties.
    *   **Machine Learning Model:** We build a neural network and train it to improve our predictions.
    *   **Profit Integration:** We change our model to take into account the profit.
    *   **Evaluation and Comparison:** We compare the different models.
    * **Visualization**: We use several visualizations to help us.
    *   **Goal:** This notebook explores different methods to predict the sex of abalones and optimizes our predictions to increase profit.

2.  **`From_prediction_to_action_inventory.ipynb`:**

    *   **Bayesian Forecasting:** We use PyMC to forecast the future demand for abalone. This is important for inventory management.
    *   **Wagner-Whitin Algorithm:** We implement the Wagner-Whitin algorithm using Dynamic Programming. This algorithm helps us decide when and how much abalone to order to minimize costs.
    *   **Markov Decision Process (MDP):** We create a Markov Decision Process to model the diver's decision-making process during a dive. The goal is to find the best strategy (collect, analyze, or ignore) to maximize profit, taking into account the time, the energy, and the penalties.
    *   **Integration:** We combine the demand forecast from PyMC with the Wagner-Whitin algorithm. We then have a way to make decisions based on our predictions.
    * **Visualization**: The optimal policy is visualized with a Gantt chart.
    *   **Goal:** This notebook shows how to combine forecasting and dynamic programming to make better decisions about inventory and harvesting. It also shows how to take into account the diver's actions.

###  Short set up for advanced users

To run the notebooks in this project, you'll need to install the following Python libraries:

*   **pandas:** `pip install pandas`
*   **numpy:** `pip install numpy`
*   **matplotlib:** `pip install matplotlib`
*   **seaborn:** `pip install seaborn`
*   **scipy:** `pip install scipy`
*   **scikit-learn:** `pip install scikit-learn`
* **tqdm**: `pip install tqdm`
* **torch**: `pip install torch`
* **pymc**: `pip install pymc`
* **arviz**: `pip install arviz`

You can install these using `pip` or `conda`.

**Running the Notebooks:**

1.  Clone the repository: `git clone [repository URL]`
2.  Navigate to the project directory.
3.  Open and run each notebook in order using Jupyter Notebook or JupyterLab.

## Key Concepts

Here are some of the main ideas explored in this project:

*   **Sustainable Harvesting:** Harvesting in a way that protects the environment and future populations.
*   **Exploratory Data Analysis (EDA):** Understanding data through visualization and basic statistics.
*   **Skewed Normal Distribution:** A statistical distribution that is useful for data that is not symmetrical.
*   **Machine Learning:** Using machine learning to improve predictions.
* **Generative Model**: A simple model to generate data.
*   **Bayesian Forecasting:** Using a probabilistic method to make forecasts, taking uncertainty into account.
*   **Markov Decision Process (MDP):** A way to model decisions over time, where the future depends on the current action.
*   **Dynamic Programming (DP):** A method for solving complex problems by breaking them down into smaller ones.
*   **Value Iteration and Policy Iteration:** Two Dynamic Programming algorithms.
*   **Wagner-Whitin Algorithm:** A dynamic programming method for inventory optimization.
* **Threshold optimization**: A way to optimize the decision threshold.
* **Profit-based decisions**: A way to choose the decision based on the profit.
* **Inventory management**: A way to manage the inventory in an optimal way.
## Further Development

This project is a good start, but there are many ways to make it even better:

*   **Real-time Data:** Add real-time data from sensors.
* **More complex models**: Use more complex models.
*   **More Complex States:** Add more variables to the MDP (e.g., the diver's energy).
*   **Stochastic Rewards:** Allow the profit to be random.
* **Computer Vision**: Use computer vision to improve the predictions.
*   **User Interface:** Create a way for the divers to use the models.
* **Deployment**: Deploy the models.
*   **Sensitivity Analysis:** See how the models change when the parameters change.
* **Generalisation**: Apply to other problems.

## Contact

Simon Hirlaender (simon.hirlaender(at).plus.ac.at)


## Installation and Setup

This project requires Python 3.11 and several Python libraries. You can set up your environment using either **Anaconda** (recommended) or a **direct Python installation**.

### Option 1: Using Anaconda (Recommended)

Anaconda is a popular distribution of Python that makes it easy to manage different environments and packages.

1.  **Install Anaconda:**
    *   If you don't already have Anaconda, download and install it from the [official Anaconda website](https://www.anaconda.com/products/distribution). Choose the version that matches your operating system (Windows, macOS, or Linux).
2.  **Create a New Environment:**
    *   Open your terminal or command prompt.
    *   Create a new Anaconda environment named `bessere_entscheidungen_mit_ki` with Python 3.11.5:
        ```bash
        conda create --name bessere_entscheidungen_mit_ki python=3.11.5
        ```
    *   Activate the new environment:
        ```bash
        conda activate bessere_entscheidungen_mit_ki
        ```
3.  **Install Packages:**
    *   Navigate to the project directory in your terminal.
    *   Use `pip` to install the required packages from the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```
    * This command will install all packages, and the correct version, that are listed in the `requirements.txt` file.

### Option 2: Direct Python Installation

If you prefer to use a direct Python installation (without Anaconda), follow these steps:

1.  **Install Python 3.11.5:**
    *   If you don't have Python 3.11.5, download it from the [official Python website](https://www.python.org/downloads/).
    *   Make sure to select the correct installer for your operating system.
    * Make sure it is 3.11.5.
2.  **Create a Virtual Environment (Recommended):**
    *   Creating a virtual environment is highly recommended to isolate the project's dependencies.
    *   Open your terminal or command prompt.
    *   Navigate to the project directory.
    *   Create a virtual environment (e.g., named `bessere_entscheidungen_mit_ki`):
        *   On Windows:
            ```bash
            python -m venv bessere_entscheidungen_mit_ki
            ```
        *   On macOS/Linux:
            ```bash
            python3 -m venv bessere_entscheidungen_mit_ki
            ```
    *   Activate the virtual environment:
        *   On Windows:
            ```bash
            bessere_entscheidungen_mit_ki\Scripts\activate
            ```
        *   On macOS/Linux:
            ```bash
            source bessere_entscheidungen_mit_ki/bin/activate
            ```
3.  **Install Packages:**
    *   Once the virtual environment is active, use `pip` to install the required packages:
        ```bash
        pip install -r requirements.txt
        ```

### Common Steps After Setting Up the Environment

1.  **Navigate to the project directory:**
    ```bash
    cd /path/to/your/project
    ```
2.  **Open the Notebooks:**
    *   If you installed the `jupyter` package (it is in the `requirements.txt`), you can run:
        ```bash
        jupyter notebook
        ```
    *   This will open Jupyter Notebook in your web browser.
3. **Open and run the notebooks.**

After completing these steps, you will have a working environment ready to run the code in this project.