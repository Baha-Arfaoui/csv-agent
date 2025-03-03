# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os

import base64
import gc
import random
import tempfile
import time
import uuid
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import streamlit as st

df1 = pd.read_csv("data/pred-app-mef-dhup-treated.csv", sep=",", encoding="utf-8")
df2 = pd.read_csv("data/pred-app3-mef-dhup-treated.csv", sep=",", encoding="utf-8")
df3 = pd.read_csv("data/pred-app12-mef-dhup-treated.csv", sep=",", encoding="utf-8")
df4 = pd.read_csv("data/pred-mai-mef-dhup-treated.csv", sep=",", encoding="utf-8")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

st.set_page_config(page_title="CSV Agent",layout="wide")

@st.cache_resource
def load_llm():
    # llm = Ollama(model="llama3.2", request_timeout=120.0)
    llm=OpenAI(model="gpt-4o-mini",)
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()



@st.cache_resource
def load_agent():
    template="""  
    Tu es un assistant expert en analyse de données CSV. Tu disposes de 5 bases de données contenant des informations sur les loyers et les données démographiques par commune en France. Utilise ces informations pour répondre précisément aux questions et réaliser des analyses. Voici la description des jeux de données :

    1. **Indicateurs de loyers (2018-2023)** : Données issues des plateformes Leboncoin et Groupe SeLoger, fournissant les loyers charges comprises par commune pour différents types de biens :
    - **Appartement (tous types)** : surface de 52 m², surface moyenne par pièce de 22,2 m²  
    - **Appartement T1-T2** : surface de 37 m², surface moyenne par pièce de 23,0 m²  
    - **Appartement T3 et plus** : surface de 72 m², surface moyenne par pièce de 21,2 m²  
    - **Maison** : surface de 92 m², surface moyenne par pièce de 22,4 m²  

    2. **Données démographiques par commune** : Informations agrégées incluant :
    - Nombre de ménages avec 1 enfant de moins de 3 ans  
    - Nombre de ménages pacsés  
    - Autres indicateurs démographiques  

    **Instructions :**  
    - Comprends bien les structures des données pour fournir des réponses précises.  
    - Si l'utilisateur pose une question, sélectionne les données les plus pertinentes parmi ces jeux.  
    - Propose des analyses ou visualisations adaptées si nécessaire.  

    
    Réponds de manière claire, concise et basée sur les données et bien formaté en markdown.  


    """


    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4o"),
        ["data/pred-app-mef-dhup-treated.csv","data/pred-app3-mef-dhup-treated.csv","data/pred-app12-mef-dhup-treated.csv","data/pred-mai-mef-dhup-treated.csv"],
        encoding="latin1",
        verbose=True,
        prompt=template,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
    return agent



        
            

# setup llm & embedding model

llm=load_llm()
agent = load_agent()


# Affichage du logo et du texte d'accueil dans la sidebar
with st.sidebar:

    st.markdown(
        """
        \n \n
        **Bienvenue !** \n
        Cette interface de chat vous permet de poser des questions sur les dataframes fournis.\n\n
     
        **Exemples :**
        - Quel est le loyer prédit par mètre carré pour Monpezat ?
        - Quelle est l'intervalle de prédiction inférieure pour Fontguenand ?
        - Analysez la relation entre le nombre d'observations et la précision du modèle pour les communes de Hirtzfelden et Oberentzen. Quelle commune a une meilleure précision du modèle par rapport au nombre d'observations ?
        - Comparez les valeurs ajustées du R-carré pour les communes de Dancevoir et Buncey. Quelle commune a un modèle plus fiable et de combien ?
        """
    )
    # Déplacer le bouton "Clear" à la fin du chat
    with st.container():
        st.markdown('<div class="clear-button">', unsafe_allow_html=True)
        if st.button("Clear Chat 🔄"):
            reset_chat()
        st.markdown('</div>', unsafe_allow_html=True)
         # Expander for Audience DataFrame
    with st.expander("Dataframes"):
        st.dataframe(df1)
        st.dataframe(df2)
        st.dataframe(df3)
        st.dataframe(df4)




   
     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with your CSV Data 🏷️")

with col2:
    st.button("Clear 🔄", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("Let's Chat !"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = agent.run(prompt)
        
        for chunk in streaming_response:
            full_response += chunk

            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})