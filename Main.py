from pathlib import Path
from textblob import TextBlob
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm

PAGE_TITLE = "Data Engineer, Educator Analyst and Technology Enthusiast"

PAGE_ICON = ":chart_with_upwards_trend:"

# Set the title and icon of the application
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# Set the path for the home page, csv file and css file
current_dir = Path(__file__).parent if "_file_" in locals() else Path.cwd()
home_page = current_dir / "Home_Page.py"
scentiment_image = current_dir / "assets" / "images" / "Scentiment.jpg"
css_file = current_dir / "styles" / "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# button position starting at false
button_clicked = False

# proivide a title for the page
st.markdown(
    """
    <h1>Interactive Sentiment Analysis</h1>
    """, unsafe_allow_html = True
)

st.markdown("---")

#open sentiment for page
image = Image.open(scentiment_image)
st.image(image)

# proivide infromation on sentiment analysis
st.markdown(
    """
    <h2>Project Overview</h2>
    <p>In this project, I will be creating a sentiment analysis application that allows users to input text and 
    receive information in a range of charts, analysing the sentiment. The application will determine the overall 
    sentiment of the text and determine if it is positive, negative, or neutral. To achieve this, I will use natural 
    language processing techniques and a pre-trained machine learning algorithm to conduct this work within the 
    project, explaining what is being shown within the charts and visuals.</p>
    <p>In addition to the sentiment analysis, this project will also include a visual 
    representation of the top 10 keywords in the input text that will provide insights into what topics might be 
    being discussed.</p>
    <p>Another aspect of the project is the calculation of a polarity, subjectivity score and sentiment itself. 
    The polarity score will provide a numeric representation of the sentiment analysis (-1 to 1) while the 
    subjectivity score will indicate the degree of subjectivity in the text. This will provide a more nuanced 
    understanding of the sentiment in the text.</p>
    <p>Finally, the project will also include a named entity recognition element, which will identify and extract 
    named entities such as people, organisations, and locations from the input text. This information can be useful 
    in understanding who or what is being discussed in the text and provides additional context for the sentiment 
    analysis.</p>
    <p>Overall, this project will provide a great opportunity to demonstrate my skills in machine learning, natural 
    language processing, and will provide valuable insights to you, the user who provides text data to be analysed.<p>
    """, unsafe_allow_html=True
    )

st.markdown("---")

# proivide infromation on how to use the tool
st.markdown(
    """
    <h2>Interactive Sentiment Analysis - How to use</h2>
    <p>To use this sentiment analysis tool, simply provide a body of text below that you would like to have analysed 
    and press the "Analyse" button. This tool will then provide you with an overall sentiment analysis of the text, 
    indicating if the sentiment is positive, negative, or neutral and a range of visuals.</p>
    """, unsafe_allow_html=True
)

with st.form(key='nlpForm'):
    # provide text area to input information
    raw_text = st.text_area("Enter Text Here")  
    # provide a button to analyse input information
    submit_button = st.form_submit_button(label='Analyse')
    # if text dataset is too small
    if not raw_text:
    # provide a warning if text dataset is too small
        st.warning("Please provide text to run sentiment analysis")
    # if button is submitted
    if submit_button:
        # devise analysis, polarity and subjectivity 
        if raw_text:
            analysis = TextBlob(raw_text)
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            # identify if polarity is positive, neutral or negative
            if polarity > 0:
                result = "Positive"
            elif polarity == 0:
                result = "Neutral"
            else:
                result = "Negative"

        # subtitle
        st.markdown(
        """
        <h2>Your Sentiment Analysis Results</h2>
        """, unsafe_allow_html = True
        )   
        # provide an information box to introduce results
        st.info("Please See Your Results Below")

        st.markdown("---")

        # provide information related to polarity & subjectivity
        st.markdown(
        """
        <h2>Polarity & Subjectivity - Data Frame Results</h2>
        <p>Polarity is a metric that is commonly used in sentiment analysis, which is a subfield of natural 
        language processing (NLP) that involves analysing the sentiment or emotion expressed in text.<p> 
        <p>Polarity refers to the positive, negative, or neutral nature of a piece of text.<p>
        """, unsafe_allow_html=True
        )

        # provide an information box to introduce polarity score, subjectivity score and sentiment
        st.info("Below you will see the 'Polarity Score', 'Subjectivity Score' and 'Sentiment")

        # provide polarity title result
        st.write("**Polarity:**")

        # provide polarity result
        st.info(polarity)

        # provide information related to polarity
        st.markdown("""
        <p>In sentiment analysis, polarity is typically measured on a scale from -1 (very negative) to 1 (very positive). 
        For example, a statement such as 'I love this product' would have a high positive polarity, while a 
        statement such as 'I hate this product' would have a high negative polarity. A statement that is neutral in 
        nature, such as 'The product is okay', would have a polarity value close to 0.<p>
        """, unsafe_allow_html=True
        )

        # provide subjectivity title result
        st.write("**Subjectivity:**")
        
        # provide subjectivity result
        st.info(subjectivity)
    
        # provide information related to subjectivity
        st.markdown("""
        <p>A subjectivity score is a value between 0 and 1 which indicates the subjectivity of a text. 
        A score of 0 means that the text is completely objective, meaning it is a fact or a statement of objectivity. 
        A score of 1 means the text is completely subjective, meaning it is an opinion or an expression of feeling. 
        A score between 0 and 1 means that the text is partially subjective and partially objective. 
        The subjectivity score is calculated based on the words used in the text, their frequency, 
        and their context.<p>
        """, unsafe_allow_html=True
        ) 

        st.markdown("---")

        # provide subjectivity result
        st.write("**Sentiment:**", result)

        # provide information related to sentiment types
        st.markdown("""
        Positive sentiment refers to a positive emotion such as happiness, excitement, or admiration. 
        <br><br>
        Negative sentiment refers to negative emotions such as anger, frustration, or sadness. 
        <br><br>
        Neutral sentiment refers to an absence of strong emotions, neither positive or negative.
        """, unsafe_allow_html=True
        )

        st.markdown("---")

        # subtitle
        st.markdown("""
        <h2>Key Words - Top 10</h2>
        """, unsafe_allow_html=True
        )
        
        # create get_word_frequency function
        def get_word_frequency(text):
            stop_words = set(stopwords.words('english'))
            words = re.findall(r'\b\w+\b', text.lower())
            words = [word for word in words if word not in stop_words]
            word_counts = Counter(words)
            return word_counts

        # create plot_top_keywords function
        def plot_top_keywords(word_counts, N=10):
            sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [item[0] for item in sorted_word_counts[:N]]
            keyword_counts = [item[1] for item in sorted_word_counts[:N]]  
            plt.bar(top_keywords, keyword_counts)
            plt.xlabel('Keywords')
            plt.ylabel('Frequency')
            plt.title('Top 10 Keywords')
            plt.xticks(rotation=45)
            return plt

        # provide information on keyword extraction
        st.markdown("""
        <p>Keyword extraction is the process of automatically identifying the most important words in a piece of text. 
        It is used to extract the most relevant and significant words from a text document. 
        The extracted keywords can then be used for various purposes, such as text classification, document clustering, 
        information retrieval, and sentiment analysis.<p>
        """, unsafe_allow_html=True
        )

        # provide an information box to support understanding the bar chart
        st.info("Below you can find a Bar Chart with the top 10 keywords identified within the text.") 

        # assign word_counts
        word_counts = get_word_frequency(raw_text)
        
        # implement error handling and plot plot_top_keywords
        if len(raw_text) < 10:
            st.error("Error: Not enough text to see Keywords Visual.")
        else:
            st.pyplot(plot_top_keywords(word_counts))        

        # provide information on keywords 
        st.markdown("""
        <p>The bar chart of keywords will often have a higher frequency of keywords on the left side compared to 
        the right side because it represents the frequency of the keywords. The x-axis of the chart is sorted in descending order, 
        meaning the keywords with the highest frequency will be on the left side, and the keywords with the lower frequency will be 
        on the right side. This is a common visual representation used to show the distribution of the keywords, and it helps to 
        identify the most frequent keywords within a text.<p>
        """, unsafe_allow_html=True
        )

        st.markdown("---")

        # create named_entity_recognition function
        def named_entity_recognition(text):
                doc = nlp(text)
                entities = []
                for ent in doc.ents:
                    entities.append((ent.text, ent.label_))
                return entities

        # create add_ner_element function
        def add_ner_element(text):
            entities = named_entity_recognition(text)
            st.write("Named Entities:")
            for entity in entities:
                st.write(entity)

        # subtitle
        st.markdown("""
        <h2>Named Entity Recognition (NER)</h2>
        """, unsafe_allow_html=True
        )
        
        # provide information on named entity recognition
        st.markdown("""
        <pNamed Entity Recognition (NER) is a process of extracting named entities such as person names, 
        organizations, locations, dates, and other predefined categories from text. The purpose of NER is to classify the named 
        entities into predefined categories, such as person names, organizations, locations, medical codes, time expressions, 
        quantities, monetary values, percentages, etc.<p>
        """, unsafe_allow_html=True
        )

        # provide an information box to support named entity introduction
        st.info("Below you can find extracted named entities from the inputted text.") 

        # error handling
        if len(raw_text) < 10:
            st.error("Error: Not enough text to conduct Named Entity Recognition.")
        else:
            add_ner_element(raw_text)

        # provide information on NER categories
        st.markdown(
                    """
                Here are some examples of NER categories:
                <ul>
                <li><b>PERSON:</b> People, including fictional.</li>
                <li><b>NORP:</b> Nationalities or religious or political groups.</li>
                <li><b>FAC:</b> Buildings, airports, highways, bridges, etc.</li>
                <li><b>ORG:</b> Companies, agencies, institutions, etc.</li>
                <li><b>GPE:</b> Countries, cities, states.</li>
                <li><b>LOC:</b> Non-GPE locations, mountain ranges, bodies of water.</li>
                <li><b>PRODUCT:</b> Objects, vehicles, foods, etc. (Not services.)</li>
                <li><b>EVENT:</b> Named hurricanes, battles, wars, sports events, etc.</li>
                <li><b>WORK_OF_ART:</b> Titles of books, songs, etc.</li>
                </ul>
                    """, unsafe_allow_html=True
                )