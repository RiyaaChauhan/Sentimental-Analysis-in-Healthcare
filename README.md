🚀 **In Progress** 🚀  
# Please note that this project is still under active development. Check back soon for updates and new features! 🔄📈

# Sentimental-Analysis in Healthcare

This project develops a model to analyze and classify sentiments from various sources, including patient reviews, social media posts, and
electronic health records. The goal is to categorize sentiments as positive, negative, or neutral. 

Key objectives include:
- **Processing Diverse Inputs**: Handling medical jargon, slang, and informal language.
- **Using Advanced Techniques**: Applying Natural Language Processing (NLP) and Machine Learning (ML) to improve sentiment analysis in healthcare contexts.

The project aims to evaluate how effectively sentiment analysis can manage complex and varied linguistic inputs in the medical field.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Dataset-Description](#dataset-descripton)
4. [Training-Data](#training-data)
5. [Testing-Data](#test-data)
3. [Features](#features)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Virtual Environment Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/RiyaaChauhan/Sentimental-Analysis-in-Healthcare.git
    cd Sentimental-Analysis-in-Healthcare
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - **Windows**:
      ```bash
      venv\Scripts\activate
      ```
    - **macOS/Linux**:
      ```bash
      source venv/bin/activate
      ```

4. Install the required dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. After installing dependencies, run the script:

    ```bash
    python sentiment.py
    ```
2. Follow the instructions in the script or consult the documentation for specific usage details.

## Dataset Descripton:
1. dataset_info.json:
- Number of Rows: 5,000
- Number of Columns: 2
- Column Names: 'Tweet', 'Sarcasm (yes/no)'
- Description: This dataset features tweets labeled for sarcasm. Each tweet is accompanied by a label ('yes' or 'no') indicating whether the tweet is sarcastic.

2. sentiment_dataset.csv:


3. McGill-NLP:
- **Columns:**
  - `abstract_id`: A unique identifier for each abstract.
  - `text`: The main content or text of the abstract.
  - `location`: The location or entity associated with the abstract.
  - `label`: A label or classification associated with the abstract.

  ```bash
  https://huggingface.co/datasets/McGill-NLP/medal

  ## Files
- **`data/training_data.csv`**: Contains the training data.
- **`data/testing_data.csv`**: Contains the testing data.


## Training Data

The model is trained on a diverse dataset, including:

- **Patient Reviews**: Text data from patient feedback across various healthcare platforms.
- **Social Media Posts**: Comments and posts related to health and wellness.
- **Electronic Health Records**: Structured data from patient health records, converted into textual descriptions.

The data includes various sentiment categories to enhance the model's ability to handle different types of input.

## Test Data

The model is evaluated using:

- **Validation Set**: A subset of the training data, kept aside for tuning model parameters and preventing overfitting.
- **Test Set**: A separate dataset that was not used during training, ensuring an unbiased assessment of the model's performance.

These datasets are designed to reflect real-world scenarios and help assess the model's accuracy in classifying sentiments across different contexts.

## Features

- **Sentiment Classification**: Categorizes sentiments into positive, negative, or neutral.
- **Diverse Input Handling**: Processes and analyzes data from patient reviews, social media posts, and electronic health records.
- **Medical Jargon Recognition**: Identifies and understands medical terminology and jargon.
- **Slang and Informal Language Processing**: Handles slang and informal language commonly found in social media and patient feedback.
- **NLP and ML Integration**: Utilizes Natural Language Processing (NLP) and Machine Learning (ML) techniques for accurate sentiment analysis.
- **Real-time Analysis**: Provides real-time insights and feedback based on analyzed data.
- **Customizable Models**: Allows for adjustments and improvements based on specific healthcare needs and data types.

## Contributing

We welcome contributions from the community to enhance the functionality and quality of this project. To contribute, please follow these guidelines:

1. **Fork the Repository**: Create your own fork of the repository to work on your changes.

2. **Create a Feature Branch**: Develop your changes in a separate branch. Use descriptive names for your branches to indicate the purpose of the changes.
   ```bash
   git checkout -b feature-branch-name

3. Make your changes

4. Commit your changes:
   ```bash
   git commit -m 'Add new feature'

5. Push to the branch
   ```bash
   git push origin feature-branch

6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
