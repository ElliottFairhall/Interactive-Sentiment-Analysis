# Interactive Sentiment Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](elliottfairhall-stock-analysis-tool-main-pgogm7.streamlit.app)

Interactive Sentiment Analysis is a Python application that performs sentiment analysis on user-provided text and presents the results in a range of charts and visuals. The application determines the overall sentiment of the text, calculates a polarity and subjectivity score, identifies and extracts named entities, and provides insights into the top 10 keywords discussed in the input text.

## Project Overview

In this project, I have created a sentiment analysis application that allows users to provide text through the use of an input field within a Streamlit application and receive information in a range of charts, analysing the sentiment and providing you with insights and a rudimentary demonstration into the possible uses of sentiment analysis on a array of texts. 

To achieve this, I used natural language processing techniques, also known as NLP, and a pre-trained machine learning algorithm to conduct this work within the project, explaining what can be achieved within the charts and visuals.

In addition to the sentiment analysis, this project includes a visual representation of the top 10 keywords in the input text, which could provide insights into what topics might be being discussed and/ or what those themes might be.

Another aspect of the project is the calculation of a polarity, subjectivity score and sentiment itself. The polarity score provides a numeric representation of the sentiment analysis (-1 to 1) while the subjectivity score indicates the degree of subjectivity in the text. This provides a more nuanced understanding of the sentiment in the text.

Finally, the project also includes a named entity recognition element, which identifies and extracts named entities such as people, organisations, and locations from the input text. This information can be useful in understanding who or what is being discussed in the text and provides additional context for the sentiment analysis.

## Requirements

The code requires the following modules to be installed:

-   `pathlib`
-   `textblob`
-   `Pillow`
-   `matplotlib`
-   `pandas`
-   `vaderSentiment`
-   `streamlit`
-   `collections`
-   `re`
-   `nltk`
-   `spacy`

In addition to the modules, the code also requires the following data to be downloaded:

-   `nltk` stopwords (`stopwords`)

The code is designed to run on Python 3.x.

The code requires the following hardware:

-   A computer capable of running Python 3.x with enough processing power and memory to run the required modules and processes.

The code requires the following operating system:

-   The code should be able to run on any operating system capable of running Python 3.x.
## How to Use

To use this sentiment analysis tool, simply provide a body of text in the text area and press the "Analyse" button. The tool will then provide an overall sentiment analysis of the text, indicating if the sentiment is positive, negative, or neutral and a range of visuals.

## Business Use

Interactive Sentiment Analysis can be used by businesses to monitor and analyse customer sentiment towards their products or services. The tool can provide insights into the positive and negative aspects of the business and help identify areas for improvement. Additionally, the named entity recognition element can help businesses identify key influencers and stakeholders in their industry.
