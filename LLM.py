import logging
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from firebase_admin import credentials, firestore
import firebase_admin
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

API_KEY = '8cf2f681b1844328953be661ba7b5ca3'
URL_API = 'https://newsapi.org/v2/everything'
 
def initialize_firebase(cred_path="mongodb-16366-firebase-adminsdk-fbsvc-bd4f5fd49e.json"):
    """
    Inicializa la conexión con Firebase usando un archivo de credenciales.
    
    Parámetros:
    - cred_path (str): Ruta del archivo JSON con las credenciales de Firebase.
    
    Retorno:
    - firestore.Client: Objeto de base de datos de Firebase.
    """
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        logging.info("Firebase inicializado correctamente.")
        return firestore.client()
    except Exception as e:
        logging.error(f"Error al inicializar Firebase: {e}")
        raise

def initialize_model(model_name='all-MiniLM-L6-v2'):
    """
    Carga un modelo de embeddings de texto usando SentenceTransformer.
    
    Parámetros:
    - model_name (str): Nombre del modelo de SentenceTransformer a cargar.
    
    Retorno:
    - SentenceTransformer: Modelo de embeddings cargado.
    """
    try:
        model = SentenceTransformer(model_name)
        logging.info(f"Modelo {model_name} cargado correctamente.")
        return model
    except Exception as e:
        logging.error(f"Error al cargar el modelo {model_name}: {e}")
        raise

def initialize_openai(api_key):
    """
    Inicializa el modelo de lenguaje de OpenAI con la clave API proporcionada.
    
    Parámetros:
    - api_key (str): Clave API de OpenAI.
    
    Retorno:
    - OpenAI: Objeto del modelo de lenguaje de OpenAI.
    """
    try:
        logging.info("OpenAI inicializado correctamente.")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return OpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0.7, memory=memory)
    except Exception as e:
        logging.error(f"Error al inicializar OpenAI: {e}")
        raise

def initialize_faiss(index_dimension=384):
    """
    Crea un índice FAISS para búsqueda vectorial de embeddings.
    
    Parámetros:
    - index_dimension (int): Dimensión de los vectores almacenados en FAISS.
    
    Retorno:
    - faiss.IndexFlatL2: Índice FAISS creado.
    """
    try:
        index = faiss.IndexFlatL2(index_dimension)
        logging.info("FAISS inicializado correctamente.")
        return index
    except Exception as e:
        logging.error(f"Error al inicializar FAISS: {e}")
        raise

def fetch_news_from_api(page_size=20, total_pages=5):
    """
    Recupera noticias de la API de NewsAPI con paginación.
    
    Parámetros:
    - page_size (int): Número de noticias por página.
    - total_pages (int): Cantidad total de páginas a recuperar.
    
    Retorno:
    - list: Lista de artículos en formato JSON.
    """
    articles = []
    for page in range(1, total_pages + 1):
        params = {
            'category': 'business',
            'apiKey': API_KEY,
            'pageSize': page_size,
            'page': page,
            'sortBy': 'publishedAt',
        }
        try:
            response = requests.get(URL_API, params=params)
            response.raise_for_status()
            articles.extend(response.json().get('articles', []))
            logging.info(f"Noticias obtenidas página {page}.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error al obtener noticias (página {page}): {e}")
            continue
    return articles

def store_news_to_firestore_and_faiss(news_articles, model, db, index):
    """
    Almacena noticias en Firestore y sus representaciones vectoriales en FAISS.
    
    Parámetros:
    - news_articles (list): Lista de noticias obtenidas de la API.
    - model (SentenceTransformer): Modelo de embeddings de texto.
    - db (firestore.Client): Base de datos Firestore.
    - index (faiss.IndexFlatL2): Índice FAISS para almacenamiento vectorial.
    
    Retorno:
    - None
    """
    for i, article in enumerate(news_articles):
        title = article.get('title', 'No title available')
        url = article.get('url', 'No URL available')
        content = article.get('description', '')

        if content:
            try:
                content_vector = model.encode([content])[0].astype(np.float32)
                news_id = f"news_{i}"
                noticia = {
                    'id': news_id,
                    'title': title,
                    'url': url,
                    'content': content,
                    'vector': content_vector.tolist()
                }
                db.collection("noticias").document(news_id).set(noticia)
                index.add(np.array([content_vector]))
                logging.info(f'Noticia almacenada con ID: {news_id}')
            except Exception as e:
                logging.error(f"Error al almacenar la noticia {i}: {e}")
        else:
            logging.warning(f'No hay contenido para el artículo: {title}')

def analyze_news_for_insights(news, llm_model=llm):
    """
    Analiza las noticias relevantes y extrae insights sobre qué sectores están más prometedores,
    usando el modelo LLM ya implementado en el sistema.
    
    Parámetros:
    - news (list): Lista de diccionarios con noticias, donde cada diccionario contiene 'title' y 'content'.
    - llm_model (OpenAI): Modelo de lenguaje utilizado para generar el análisis.
    
    Retorna:
    - str: Análisis generado por el modelo de lenguaje.
    """
    news_text = "\n".join([f"Título: {news_item['title']}\nResumen: {news_item['content']}" for news_item in news])
    prompt = f"""
    Basado en las siguientes noticias financieras, proporciona un análisis detallado sobre qué sectores están
    mostrando señales de crecimiento o decrecimiento. Además, sugiere cómo deberían ajustarse las asignaciones
    en una cartera de inversión ficticia, indicando qué sectores deberían recibir más o menos inversión.

    Noticias:
    {news_text}
    """
    response = llm.predict(prompt)  
    return response.strip()

def optimize_portfolio(assets, news, llm_model):
    """
    Optimiza la cartera de inversión en función de los insights generados a partir de noticias financieras.
    
    Parámetros:
    - assets (list): Lista de diccionarios representando los activos en la cartera, cada uno con 'name' y 'price'.
    - news (list): Lista de diccionarios con noticias financieras.
    - llm_model (OpenAI): Modelo de lenguaje para generar el análisis.
    
    Retorna:
    - list: Lista de activos con nuevas asignaciones optimizadas.
    """
    try:
        insights = analyze_news_for_insights(news, llm_model)
        print("Insights generados desde las noticias:", insights)
        sector_map = {
            "EquityTech Fund": "Tecnología",
            "GreenEnergy ETF": "Energía Renovable",
            "HealthBio Stocks": "Salud/Biotecnología",
            "Global Bonds Fund": "Bonos",
            "CryptoIndex": "Criptomonedas",
            "RealEstate REIT": "Bienes Raíces",
            "Emerging Markets Fund": "Mercados Emergentes",
            "AI & Robotics ETF": "Inteligencia Artificial y Robótica",
            "Commodities Basket": "Materias Primas",
            "Cash Reserve": "Efectivo"
        }
        allocations = {
            "EquityTech Fund": 0.20,
            "GreenEnergy ETF": 0.15,
            "HealthBio Stocks": 0.10,
            "Global Bonds Fund": 0.10,
            "CryptoIndex": 0.10,
            "RealEstate REIT": 0.10,
            "Emerging Markets Fund": 0.10,
            "AI & Robotics ETF": 0.05,
            "Commodities Basket": 0.05,
            "Cash Reserve": 0.05
        }
        
        adjustment_factors = {"incrementar": 0.05, "reducir": -0.05}
        for asset, sector in sector_map.items():
            if sector.lower() in insights.lower():
                allocations[asset] += adjustment_factors["incrementar"]
            if f"{sector.lower()} en decrecimiento" in insights.lower() or f"{sector.lower()} baja volatilidad" in insights.lower():
                allocations[asset] += adjustment_factors["reducir"]
        
        total_allocation = sum(allocations.values())
        for asset in allocations:
            allocations[asset] /= total_allocation  
        total_value = sum([asset["price"] for asset in assets])
        

        for asset in assets:
            if asset["name"] in allocations:
                asset["allocation"] = total_value * allocations[asset["name"]]
        return assets
    except Exception as e:
        logging.error(f"Error al optimizar la cartera: {e}")
        return assets

class NewsTools:
    """
    Clase que proporciona herramientas para obtener noticias y optimizar la cartera de inversión.
    """
    def __init__(self, db, model, index):
        """
        Inicializa la clase con los recursos necesarios.
        
        Parámetros:
        - db: Conexión a la base de datos.
        - model: Modelo de lenguaje para análisis de noticias.
        - index: Índice para búsqueda en la base de datos.
        """
        self.db = db
        self.model = model
        self.index = index

    def fetch_news(self, query: str, num_pages: int = 5) -> str:
        """
        Obtiene noticias financieras y las almacena en la base de datos y el índice de búsqueda.
        
        Parámetros:
        - query (str): Término de búsqueda para obtener noticias.
        - num_pages (int): Número de páginas a recuperar.
        
        Retorna:
        - str: Mensaje indicando el número de noticias obtenidas o un mensaje de error.
        """
        try:
            articles = fetch_news_from_api(page_size=20, total_pages=num_pages)
            store_news_to_firestore_and_faiss(articles, self.model, self.db, self.index)
            return f"Se han obtenido y almacenado {len(articles)} noticias."
        except Exception as e:
            logging.error(f"Error al obtener y almacenar noticias: {e}")
            return "Hubo un error al obtener y almacenar las noticias."

    def optimize_portfolio(self, assets: list, news: list, llm_model: OpenAI) -> str:
        """
        Optimiza la cartera basándose en el análisis de noticias.
        
        Parámetros:
        - assets (list): Lista de activos en la cartera.
        - news (list): Lista de noticias financieras.
        - llm_model (OpenAI): Modelo de lenguaje.
        
        Retorna:
        - str: Mensaje indicando el resultado de la optimización.
        """
        try:
            optimized_assets = optimize_portfolio(assets, news, llm_model)
            return f"Cartera optimizada: {optimized_assets}"
        except Exception as e:
            logging.error(f"Error al optimizar la cartera: {e}")
            return "Hubo un error al optimizar la cartera."

# Inicialización
def initialize_system():
    db = initialize_firebase()
    model = initialize_model()
    index = initialize_faiss()
    api_key = "tu_clave_de_api"
    llm = initialize_openai(api_key)

    return db, model, index, llm